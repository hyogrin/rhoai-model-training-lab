# rhoai-model-training-lab — τ-Knowledge 프로그램 명세와 Opus 구현 프롬프트

개정일: 2026-09-17. Revision 2. 목표: Red Hat OpenShift AI Self-Managed 3.5.x.

이 문서는 저장소 구현 명세다. 이 문서 수정 자체가 synthetic 데이터 생성, GPU 학습 또는 클러스터 배포를 실행했다는 의미는 아니다.

## 0. 이번 개정의 결정 및 사용 방법

도메인을 **τ-Knowledge의 banking_knowledge**로 확정한다. 핵심 task는 은행 고객의 요청에 대해 지식 문서와 현재 계정 정보를 확인하고, 정책에 맞게 답변하거나 시뮬레이션 도구로 업무를 수행하는 것이다.

**데이터 준비를 먼저 완성하고 학습 노트북을 단순하게 만든다.** 운영자/강사가 SDG와 검증·분할·학습 형식 변환을 한 번 수행하여 버전이 고정된 `prepared dataset bundle`을 배포한다. 참가자는 teacher API나 SDG 실행 없이 bundle을 받아 LoRA·OSFT 학습을 시작할 수 있어야 한다. 원본 수집·SDG 노트북은 별도의 제작·심화 경로로 유지한다.

| 단계 | 담당 흐름 | 결과 |
|---|---|---|
| 1A. 사전 데이터 제작 | 원본 고정 → SDG → 독립 검증 → 분할 → 형식 변환 → bundle release | 바로 학습 가능한 JSONL, manifest, 품질 보고서, 설정 |
| 1B. 참가자 학습 | bundle 다운로드·검증 → LoRA / OSFT 독립 학습 → export·배포 | 각 모델 checkpoint, serving endpoint |
| 2. RAG / harness | 같은 KB 및 학습 예제 index → planning·tool 실행 backend | 간단한 실행 스크립트, 공식 환경 연결 adapter |
| 3. 평가 | 준비된 offline 진단 + 공식 τ-Knowledge episode + 능력 유지 평가 | MLflow experiments, 비교 노트북 |

Opus에는 이 파일 전체와 10번 공통 프롬프트를 전달한다. 단계별 작업은 11~14번, 최종 검토는 15번을 사용한다. 기존 저장소가 있으면 변경 사항을 보존하고 `docs/implementation-status.md`를 읽어 이어서 구현한다.

## 1. 범위와 기본 설계

- 기존 RHOAI Workbench에서 시작한다. 클러스터·Workbench 설치는 선행조건 안내만 제공한다.
- Qwen 후보는 `Qwen/Qwen3-4B-Instruct-2507`이다. LoRA/OSFT backend와 tool-capable serving의 공통 지원을 실제 확인하고 revision을 고정한다. 더 작은 Qwen은 smoke용으로만 선택 가능하며 공식 agent task를 잘 수행한다고 보장하지 않는다.
- `sdg_hub`가 실제 synthetic 생성 경로를 담당하고, `training_hub`가 실제 LoRA/OSFT 학습을 담당한다. import만 하고 다른 구현으로 우회하지 않는다.
- 학습 분기는 동일 base checkpoint 및 동일 논리적 학습 샘플에서 독립적으로 시작한다. LoRA 결과에 OSFT를 적용하지 않는다.
- 기본 인터페이스는 텍스트다. 음성, UI, Dify, 필수 OpenShell/MCP, 별도 운영용 vector DB는 범위 밖이다.
- FastAPI + 작은 LangGraph harness + 로컬 영속 vector index + τ 공식 simulation adapter를 사용한다. 업무 도구는 공식 시뮬레이션의 구현을 연결한다.
- SDG/평가용 teacher·user simulator는 별도 endpoint를 사용한다. 참가자 학습에는 이 endpoint가 필요 없어야 한다.
- 모든 문서는 한국어로 설명하고 benchmark·학습 주요 콘텐츠는 원문 영어를 유지한다. 번역 데이터는 별도 실험이다.

## 2. 근거 자료와 확인할 구현

### 2.1 τ-Knowledge / τ-bench

[τ-Knowledge 논문](https://arxiv.org/abs/2603.04370)은 2026-03-04 공개되었으며, 은행 상담에서 비정형 지식과 도구 기반 업무 수행을 함께 평가한다. [공식 저장소](https://github.com/sierra-research/tau2-bench)의 경로는 `tau2-bench`지만 현재 README는 τ³-bench와 `banking_knowledge`를 설명한다. 이름으로 버전을 추정하지 말고 release와 commit SHA를 고정한다.

확인 시점 README는 2026년 7월 v1.0.1의 banking grading 수정과 그 전후 점수의 비호환성을 명시한다. 구현 시 그 수정이 포함된 버전을 우선 검토하고 scorer/task revision을 함께 저장한다. Python 요구사항도 확인해야 하며 RHOAI 학습 환경과 τ evaluator 환경은 분리할 수 있다.

[Knowledge Retrieval 문서](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/knowledge/README.md)는 `no_knowledge`, `full_kb`, `golden_retrieval`, BM25, dense retrieval 등의 구성을 설명한다. `golden_retrieval`은 정답 근거를 쓰는 oracle 진단 전용이다. benchmark 코드의 실제 task schema, reward, tool discovery, simulator 권한, split을 먼저 조사한다. 문서에 보이는 설정을 설치 버전에서 검증 없이 호출하지 않는다.

### 2.2 기존 참고 레포의 역할

| 참고 | 재사용할 역할 | 그대로 복사하지 않을 것 |
|---|---|---|
| [SDG Hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub) | YAML flows, structured generation, filtering, checkpoint | 존재를 확인하지 않은 내장 flow/API |
| [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) | 실제 `lora_sft`, `osft` API와 export 사례 | upstream과 제품 포함 버전이 같다는 가정 |
| [micro-financial-loan](https://github.com/cbtham/micro-financial-loan/tree/main) | LoRA merge, HF artifact 저장, S3 upload, vLLM 배포 흐름 | 고정 비밀/cluster 주소, TLS 검증 비활성화, 학습 시간 가정 |
| [rhoai-custom-research-lab](https://github.com/hyogrin/rhoai-custom-research-lab) | graph/state, context·tools·observability 계층 | 전체 UI/DB/MCP 스택의 필수화 |
| [run_eval.py](https://github.com/hyogrin/rhoai-inference-design-planner/blob/main/scripts/run_eval.py) | CLI, 결과 파일, MLflow params/metrics/artifacts | golden 문장 유사도 기반 100점 rubric |
| [RHOAI 3.5 문서](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/release_notes/index) | 실제 Workbench/serving/MLflow 구성 확인 | 자동 설치·자동 추적을 모든 custom 코드에 대한 보장으로 해석 |

기존 run_eval.py는 저장된 응답을 채점하는 코드다. 새 구현에는 실제 endpoint 호출 및 τ episode 실행·공식 reward 수집이 필요하다. 문장 유사도를 업무 성공률로 부르지 않는다.

`docs/reference-audit.md`에는 실제 읽은 source path, URL, SHA, signature, 가져온 패턴, 변경점과 상태를 기록한다. harness에서는 `agents/orchestrator/graph.py`, `state.py`, `layers/`, `backend/api.py`, `backend/observability.py` 등을 확인한다. 제품 지원, upstream 기능, 현재 환경 실측 성공을 구분한다.

## 3. 먼저 만들 산출물: 준비된 데이터 bundle

### 3.1 참가자 경로와 제작자 경로

참가자 기본 quickstart는 다음의 **구현 목표 명령**으로 끝나야 한다. 이 명세의 lab 함수/CLI 이름은 앞으로 구현할 계약이지 upstream API가 아니다.

```bash
bash scripts/setup_env.sh --profile lora
python scripts/fetch_prepared_dataset.py --release configs/data-release.yaml
python scripts/validate_prepared_dataset.py --bundle data/prepared/tau-knowledge-v1
# 이후 03 또는 04 학습 노트북 실행
```

운영자 경로는 별도로 제공한다.

```bash
python scripts/prepare_tau_sources.py --config configs/data-preparation.yaml
python scripts/generate_synthetic.py --config configs/sdg.yaml
python scripts/validate_synthetic.py --config configs/data-preparation.yaml
python scripts/build_prepared_bundle.py --config configs/data-release.yaml
```

fetch 실패 시 SDG를 조용히 실행하지 않는다. 누락 파일, hash 불일치, license 조건, 버전 차이를 명확히 보고한다. 인터넷이 없는 Workbench에서는 동일 bundle을 PVC/S3에서 불러올 수 있다.

### 3.2 Bundle 구성

| 경로 | 내용 |
|---|---|
| `manifest.json` | bundle version, 실제 τ SHA/task revision, source hash, model/tokenizer profile, counts, split policy |
| `checksums.sha256` | 모든 파일 무결성 |
| `canonical/train.jsonl`, `validation.jsonl` | 검증된 공통 chat/tool examples, 모델 독립 schema |
| `training/lora/train.jsonl`, `validation.jsonl` | 고정 LoRA backend에서 load 가능한 형식 |
| `training/osft/train.jsonl`, `validation.jsonl` | 고정 OSFT backend에서 load 가능한 형식 |
| `configs/lora.yaml`, `osft.yaml` | 데이터 경로·학습 profile; 실제 GPU에 따른 override 가능 |
| `kb/documents.jsonl` | 공개 허용된 KB snapshot, doc IDs, 내용, 링크·출처 |
| `examples/train_examples.jsonl` | train-only 검색 예제, provenance 포함 |
| `metadata/provenance.jsonl` | canonical ID, source docs, scenario family, generator/validator, 결과 |
| `metadata/splits.json` | task/scenario family 분할, official holdout ID 목록 또는 hash |
| `reports/quality.json`, `quality.md` | 유형별 통과율, reject 사유, token 통계, coverage, 검토 결과 |
| `reports/backend-validation.json` | 두 backend loader/tokenizer 검사 및 실제 수행 범위 |
| `DATASET_CARD.md`, `LICENSES/` | 생성·검증·분할·제한사항·재배포 조건 |

평가용 private instructions, expected actions, reward criteria는 **학습/RAG bundle에 포함하지 않는다**. 별도 `evaluation bundle` 또는 공식 설치 경로에 두고 evaluator만 접근한다. client-visible user message와 evaluator-private user scenario를 구분한다.

생성된 bundle은 학습 스크립트만 있는 상태가 아니라, 실제 검증된 data bytes와 checksum을 제공하는 것이 목표다. 작은 smoke bundle은 사용권이 허용하면 repo에 포함하고 큰 lab bundle은 release artifact/S3/OCI 등의 고정 URI에 둔다. remote upload 권한이 없으면 로컬 bundle까지 완성하고 정확한 배포 절차를 남긴다. 존재하지 않는 download URL이나 checksum을 만들지 않는다.

초기 제안 규모: smoke 64개, lab 2,000~5,000개의 통과 샘플. 숫자는 목표이며 보장치가 아니다. 실제 통과 수와 coverage에 따라 profile을 확정한다. 데이터 개수만 늘리기 위해 저품질 샘플을 포함하지 않는다.

## 4. SDG 설계: 생성보다 검증·학습 편의성을 먼저

### 4.1 생성 원천과 공정성

- KB snapshot, 공개 agent policy, tool schema와 별도로 제작한 학습용 simulator state를 사용한다.
- 공식 train split이 존재하고 사용 조건이 맞으면 그 task만 seed로 사용할 수 있다. 실제 split 유무를 확인하기 전 train/dev/test가 있다고 가정하지 않는다.
- 적절한 공식 train split이 없으면 **공식 task 전체를 평가용으로 보류**하고 KB와 새 scenario generator만으로 synthetic train/dev를 만든다.
- 공개 KB를 학습에 사용하는 실험은 `kb_adaptation`으로 명시한다. 이는 KB 내용을 배우고 새 업무 상황에 적용하는 실험이며 unseen-document 일반화가 아니다. 원래 benchmark의 training-free leaderboard와 동등 비교하지 않는다.
- 평가 task의 문구, hidden scenario, 정답 action sequence, golden document list, task별 evaluator state를 보고 학습 시나리오를 생성하지 않는다.
- 학습용 문서 전체가 평가 KB와 겹치는 것은 `kb_adaptation`의 의도된 조건이다. 반면 task/scenario·답변 누출은 차단한다. 기존의 무조건 document-disjoint 요구로 이 실험을 모순되게 만들지 않는다.
- 별도 `unseen_policy` 확장에서는 policy/document family를 SDG 전에 분리하고, 평가 때 모든 모델에 동일 근거 접근을 제공한다.

### 4.2 학습 유형

| 유형 | 입력 → 출력 | 검증 |
|---|---|---|
| 정책 지식 QA | 질문 → 정확한 조건·예외 설명 | 출처·조건·한도·단위 일치 |
| 문맥 기반 정책 적용 | 문서 + 고객 상황 → 허용/불가/추가 확인 | 조건 truth table 또는 검토된 구조화 규칙 |
| 추가 정보 질문 | 불완전한 상황 → 꼭 필요한 확인 질문 | 실제 누락 field와 질문의 대응 |
| 도구 선택·인자 | 현재 관측 + 사용 가능한 tools → 다음 tool call | schema, entity/state 일치 |
| 다중 단계 trajectory | 대화·tool observations → assistant actions·응답 | 공식 simulation replay + 정책·최종 상태 검사 |
| 예외·충돌·근거 부족 | 제한된 근거 → 보류/추가 검색/적절한 종료 | 근거 없는 성공·임의 조건 생성 차단 |

단순 QA만으로 학습하고 공식 agent 업무 수행이 개선될 것이라고 가정하지 않는다. 정식 lab bundle에는 검증된 tool-use 사례와 짧은 trajectories를 포함한다. 다중 단계 예제는 더 어려우므로 별도 품질 기준과 실제 통과 수를 보고한다.

분량 목표 예: 정책 QA 30%, 정책 적용 25%, 도구 선택 20%, trajectory 15%, 추가 확인·예외 10%. 실제 분포는 generator/validator가 지원하는 범위에서 확정하고 manifest에 기록한다. 이 비율은 benchmark의 공식 분포가 아니다.

### 4.3 SDG Hub pipeline

1. KB를 문서 단위로 수집하고 섹션·링크·정책 버전·조건·예외를 보존한다.
2. generator가 policy facts/conditions를 구조화한다. 원문 offset/doc ID를 남기고 자동 추출 오류를 표본 검토한다.
3. scenario template/family를 **생성 전에** train/validation으로 나눈다. paraphrase, 숫자·이름 변형, 동일 workflow의 형제 예제는 같은 split에 둔다.
4. teacher가 명확한 JSON schema로 상황·응답·필요한 tool calls를 생성한다. 실제 고객/계정 ID 대신 독립 simulation fixture를 사용한다.
5. 정적 schema, source grounding, 조건/단위, tool names/arguments, 대화 순서, citations를 검사한다.
6. tool 예제는 실제 공식 simulator adapter로 replay하여 precondition과 결과 상태를 검사한다. generator의 말만으로 성공 처리하지 않는다.
7. 독립 validator와 표본 사람 검토를 수행한다. LLM judge만으로 완전 검증되었다고 표시하지 않는다. generator와 validator가 같으면 그 사실도 기록한다.
8. exact/near-duplicate 및 scenario family 누출을 검사한다. 평가 누출 검사는 evaluator 격리 환경에서 pass/fail·hash 중심 결과만 돌려준다.
9. 공통 schema로 정규화한 후 backend별 학습 파일을 미리 생성한다.
10. 두 backend loader와 tokenizer로 모든 샘플을 검사한 뒤 bundle을 release한다.

실행 budget, concurrency, retry/backoff, resume, 거절 이유별 로그, teacher usage를 지원한다. 잘못된 자료는 별도 quarantine에 둔다. 긴 내부 사고 추적을 생성·보관하도록 요구하지 않으며 짧은 계획·근거·관측 가능한 action만 학습한다.

### 4.4 Backend-ready 형식과 loss

canonical example에는 `sample_id`, `messages`, 선택적 `tools`, `source_doc_ids`, `scenario_family`, `validation_status`가 있다. provenance는 학습 입력에서 제거하여 sidecar에 둔다.

```json
{
  "sample_id": "synthetic-policy-0001",
  "messages": [
    {"role": "system", "content": "You are a banking support assistant."},
    {"role": "user", "content": "<synthetic request and permitted observations>"},
    {"role": "assistant", "content": "<validated response>"}
  ]
}
```

위는 계약 예시이며 그대로 학습할 dummy data가 아니다. tool-call examples는 실제 모델의 chat template와 학습 backend가 허용하는 tool representation을 확인해 변환한다. OpenAI 형식의 tool message를 모든 backend가 그대로 처리한다고 가정하지 않는다.

- 같은 canonical IDs가 LoRA/OSFT 파일에 1:1로 대응해야 한다. export 과정의 omission/변경을 비교하고 hash mapping을 저장한다.
- assistant text 및 assistant tool-call targets에만 loss를 적용한다. system/user/tool observation과 padding은 mask한다. backend가 이를 지원하지 않으면 검증된 전처리 경로를 구현하거나 해당 profile을 blocked로 표시한다.
- tool schema 렌더링과 assistant call serialization은 inference template와 일치해야 한다. 필요한 tool schema를 누락한 학습 예제를 만들지 않는다.
- template는 정확히 한 번만 적용한다. BOS/EOS/pad, assistant 종료, tool_call_id 참조와 순서를 점검한다.
- default는 role messages JSONL이다. token cache는 선택이며 tokenizer revision/template hash가 다르면 다시 만들어야 한다.
- 모델/tokenizer별 profile은 최대 길이와 실제 token 분포를 미리 검증한다. 긴 tool observation은 근거와 상태를 유지하며 준비 단계에서 축약하거나 제외한다. 학습 notebook에서 임의 truncation하지 않는다.
- 두 backend가 다른 storage 형식을 요구하더라도 logical samples와 loss 대상 의미는 같아야 한다. 유지 불가하면 공정성 제한을 보고한다.

## 5. 단순한 학습 노트북과 배포

### 5.1 실행 경로

| 경로 | 노트북 | 역할 |
|---|---|---|
| 참가자 | `00_preflight.ipynb` | GPU·환경·PVC·MLflow·model profile 확인 |
| 참가자 | `01_load_prepared_dataset.ipynb` | bundle 가져오기, checksum·호환성, 예제·통계 보기 |
| 참가자 | `03_lora_finetuning.ipynb` | 이미 준비된 파일로 학습·저장 |
| 참가자 | `04_osft_finetuning.ipynb` | 같은 base/data로 독립 학습·저장 |
| 참가자 | `05_export_and_deploy.ipynb` | LoRA merge/OSFT export, S3, serving smoke |
| 참가자 | `06_rag_harness.ipynb` | index·planning·tool adapter·API 실습 |
| 참가자 | `07_evaluate.ipynb` | offline/episode 평가 및 MLflow |
| 참가자 | `08_compare_results.ipynb` | 실험 결과 조회·비교 |
| 운영자/심화 | `data_preparation/01_prepare_tau_sources.ipynb` | KB/task boundary와 source 고정 |
| 운영자/심화 | `data_preparation/02_generate_synthetic.ipynb` | SDG Hub flow 설명·실행 |
| 운영자/심화 | `data_preparation/03_validate_and_release.ipynb` | 품질 검증, backend export, release |

03/04의 사용자 실행 셀은 각각 대략 **5~7개**를 목표로 한다: 설정 → bundle 검사 → sample/loss preview → 학습 실행 → 결과/MLflow → checkpoint reload/export 안내. 설명 셀은 충분히 제공한다.

학습 노트북에는 source download, teacher inference, duplicate 제거, split, 복잡한 tool-format 변환, 데이터 수작업 수정이 없어야 한다. 데이터 문제가 발견되면 재생성 대신 bundle validation 오류로 중단한다. 심화 노트북으로 연결한다.

교육 목적상 실제 `training_hub.lora_sft` / `osft` 호출과 주요 인자는 짧은 셀 또는 연결된 source로 보여준다. 세부 호환성 코드는 얇은 adapter로 격리한다. 기능을 감춘 거대한 자체 framework를 만들지 않는다.

### 5.2 학습 조건

- base revision, tokenizer profile, bundle ID, canonical sample IDs, seed를 동일하게 고정한다.
- LoRA 초기 r=16/alpha=32, OSFT unfreeze rank ratio 0.25는 출발점이며 실제 지원과 dev 결과로 확정한다. learning rate를 두 기법에 억지로 동일하게 만들지 않는다.
- 데이터 규모, tuning budget, 실제 처리 tokens/steps, wall time, peak VRAM을 기록한다. 동일 epoch가 동일 compute는 아니다.
- OSFT 분해·activation·optimizer 메모리를 고려한다. 단순 가중치 크기만으로 자원을 추정하지 않는다.
- SDG, LoRA, OSFT, τ evaluation은 필요하면 별도 environment/kernel을 사용한다. torch/CUDA 덮어쓰기 설치를 피하고 lock/constraints를 제공한다.
- 학습은 subprocess 등으로 분리하고 GPU를 순차 사용한다. checkpoint resume와 실패 로그를 남긴다.
- bundle 품질이 학습 자원·모델 용량 문제를 없애지는 않는다. 작은 모델의 full benchmark 성공률을 사전 보장하지 않는다.
- full SFT는 선택 대조군이다. OSFT의 일반 SFT 대비 인과적 망각 완화 주장은 해당 대조군 없이 하지 않는다.

### 5.3 RHOAI 3.5 배포

실제 설치 CRD/runtime template와 공식 3.5 문서를 확인해 single-model vLLM 배포 경로를 사용한다. MaaS/llm-d는 필수가 아니다. runtime image digest, 모델 revision, chat template와 tool parser 설정을 기록한다.

LoRA adapter를 보존하고 merge된 HF 모델/tokenizer/config를 별도 export한다. OSFT checkpoint도 serving 가능한 HF export로 변환·reload한다. shard/index/파일 hash와 tool-call 응답을 검사한다. S3 호환 업로드를 기본으로, ModelCar는 선택으로 제공한다. 모델 가중치와 runtime 이미지를 구분한다.

기존 예제의 고정 credentials와 TLS 비활성화를 복사하지 않는다. endpoint별 Secret/CA/config를 사용한다. 권한이 없으면 manifest/render·실행 명령까지 작성하고 실제 apply 상태를 구분한다. namespace 범위 cleanup, ready 검사, 인증된 chat 및 tool-call smoke를 제공한다.

GPU 하나에서 base/LoRA/OSFT를 순차 서빙할 수 있게 한다. 같은 endpoint를 재사용하면 매 평가 전 model artifact hash가 일치하는지 검사한다. `/v1` 중복, served model name, tool_call parsing 오류를 preflight에서 확인한다.

## 6. RAG와 planning harness

### 6.1 지식·상태·평가 정보의 경계

| 영역 | 역할 | agent 접근 |
|---|---|---|
| KB snapshot | 정책·상품·업무 절차 | 설정에 따라 검색 또는 문맥 |
| synthetic train examples | 문제 해결·도구 사용 예시 | train-only example index 옵션 |
| 현재 simulator DB | 해당 episode의 계정 상태 | 공식 도구를 통해 허용된 관측만 |
| private task criteria | 성공 조건·expected actions·hidden user goals | evaluator만 |

업무 DB 전체를 prompt나 vector index에 넣지 않는다. user simulator private instructions를 assistant request로 전달하지 않는다. 공식 tool discovery 메커니즘을 존중한다. lab convenience mode에서 모든 schema를 미리 노출하면 benchmark와 다른 조건임을 표시한다.

### 6.2 실행 구조

FastAPI, 작은 LangGraph state graph, 로컬 persistent vector store를 사용한다. BM25는 진단 baseline으로 유지하고 dense index는 고정 embedding 모델을 이용한다. 로컬 CPU embedding 또는 지정된 endpoint를 선택 가능하게 한다. τ 기본 Qwen embedding config가 RHOAI endpoint를 자동으로 사용한다고 가정하지 말고 adapter를 만든다.

KB collection과 example collection을 분리한다. example은 방법의 참고이며 현재 계정의 사실 근거가 아니다. index hash에는 corpus, embedding revision/dimension, chunking version을 포함한다. 정책 예외와 문서 링크가 잘리지 않게 chunking한다.

graph: 관측 수신 → 짧은 구조화 plan → 지식 검색/도구 발견 → 정책 적용 → 허용된 업무 tool call → 결과 관측·검증 → 필요한 질문 또는 다음 단계 → 완료/실패.

- planner의 결과가 실제 query/tool/분기를 바꾸어야 한다.
- 최대 turn, tool call, repair, token, wall-clock budget을 둔다. user-facing 질문은 공식 simulator 또는 실제 사용자의 응답을 기다린다. 답을 스스로 만들어 진행하지 않는다.
- verifier는 현재 관측과 공개 policy만 본다. hidden expected action/state를 참조하지 않는다.
- state-changing tool call은 무조건 재시도하지 않는다. 공식 semantics와 결과 관측으로 중복 실행을 방지한다.
- 동일 task/trial 시작 시 환경을 초기화하고 session별 state를 격리한다. 병렬 평가 시 DB와 simulator를 공유하지 않는다.
- `simple_rag`는 추가 planning/repair 없이 각 대화 turn에서 고정 검색 후 응답/다음 action을 만드는 비교 모드다. `agent_rag`는 같은 모델·retriever·도구로 planning과 추가 검색을 허용한다.
- 기본은 simulation 전용이다. 실제 은행 계정이나 외부 업무 시스템과 연결하지 않는다.

### 6.3 API와 실행 스크립트

`GET /healthz`, `GET /readyz`, `POST /v1/agent/step`, `POST /v1/qa`를 제공한다.

`/v1/agent/step`은 session/request ID, 허용된 message history, tool observations, 현재 노출된 schemas, mode를 받아 다음 assistant message 또는 tool calls를 반환한다. 도구 실행/상태 갱신은 명시된 simulation adapter가 소유한다. hidden task ID 조회로 정답을 찾는 경로를 만들지 않는다.

`/v1/qa`는 준비된 지식 QA 진단용이며 공식 업무 성공을 대신하지 않는다. 응답에는 status, content/tool_calls, citations(적용 시), model/artifact/corpus hash, usage, trace ID를 넣는다. 401/403/timeout을 성공 응답으로 감싸지 않는다.

```bash
bash scripts/setup_env.sh --profile backend
bash scripts/build_index.sh --bundle data/prepared/tau-knowledge-v1
bash scripts/start_backend.sh --host 127.0.0.1 --port 8000
bash scripts/smoke_backend.sh
bash scripts/stop_backend.sh
```

스크립트는 foreground 기본, 선택 background의 PID/log 관리, port 충돌 검사, 소유 PID만 종료를 지원한다. sudo/systemd/Docker-in-Docker는 요구하지 않는다. Containerfile 및 Deployment/Service/Route/PVC/Secret 예시에서 임의 UID·쓰기 권한·probe·resources를 고려한다. shell 검색 sandbox는 선택이며 필수로 추가하지 않는다.

## 7. 평가: 쉬운 진단과 공식 업무 성공을 분리

### 7.1 세 가지 track

**A. `prepared_diagnostics` — 노트북에서 이해하기 쉬운 평가**

별도 생성·검토한 holdout QA, 정책 적용, 다음 action 예제를 사용한다. canonical task family를 train과 분리하고 label을 evaluator에만 둔다. 정답/조건 일치, tool name/arguments, 근거 일치, 추가 확인 필요 여부를 채점한다. 가능한 항목은 deterministic check, 나머지는 명시된 rubric·고정 judge·표본 검토를 사용한다.

이 점수는 **τ-Knowledge-derived lab diagnostic**이다. synthetic 단일 턴 정답률을 공식 τ pass rate라고 부르지 않는다. `matched_context` 조건에서는 Base/LoRA/OSFT에 동일 KB excerpt와 observations를 제공한다.

**B. `tau_episodes` — 공식 환경에서 end-to-end 평가**

공식 runner 또는 의미가 보존된 runner adapter가 user simulator, tools, 환경 초기화, 종료 및 grading을 담당한다. 매 assistant turn의 모델 추론은 실제 배포된 endpoint를 호출한다. backend를 사용하는 경우 공식 agent interface와 `/v1/agent/step`을 연결한다. 문자열 답변만 보고 task 성공을 판단하지 않는다.

기본 비교는 동일한 controller/tool 환경 아래 `model × knowledge access` 요인으로 분리한다.

| variant | 모델 | 지식 접근 |
|---|---|---|
| base_no_knowledge | Base | KB retrieval 없음 |
| lora_no_knowledge | LoRA | KB retrieval 없음 |
| osft_no_knowledge | OSFT | KB retrieval 없음 |
| base_rag | Base | 동일 RAG |
| lora_rag | LoRA | 동일 RAG |
| osft_rag | OSFT | 동일 RAG |

현재 계정 상태, 업무 tools, simulator 조건은 전 군 동일하다. `no_knowledge`는 업무 도구 제거를 의미하지 않는다. `kb_adaptation`에서 학습과 검색에 사용한 KB scope와 실제 학습 coverage를 보고한다.

planning 효과는 `base_simple_rag` 대 `base_agent_rag`로 별도 비교한다. example retrieval 효과는 동일 설정에서 `use_examples=false/true`로 비교한다. 모델 학습 효과와 controller/example/도구 차이를 하나의 원인으로 해석하지 않는다.

Task IDs와 trials, simulator model/revision/prompt/seed, budgets, model sampling, retriever 설정을 고정한다. 공개 eval task를 hyperparameter 탐색에 반복 사용하지 않는다. 별도 synthetic dev를 사용한다. 원래 task/split/정책을 변경하면 official score가 아닌 modified protocol로 보고한다.

τ의 `pass^k`는 반복 수행의 일관된 성공을 나타내는 지표이며 코드 생성의 best-of-k `pass@k`와 다르다. 정확한 estimator는 고정 공식 evaluator를 사용한다. trial 수가 부족하면 해당 metric은 계산하지 않는다. action recall은 보조 지표이고 불필요하거나 잘못된 action을 놓칠 수 있어 task success 대체물로 쓰지 않는다.

**C. `retention` — 기존 능력 유지**

KB/RAG/금융 system prompt를 끄고 고정된 별도 benchmark를 Base/LoRA/OSFT로 동일하게 평가한다. 기본은 ARC-Challenge generated-answer accuracy, 선택 확장은 공식 구현을 확인한 IFEval이다. log-likelihood leaderboard 점수와 생성형 accuracy를 혼동하지 않는다. `retention_delta_pp = 100 × (adapted_accuracy - base_accuracy)`를 기록한다. 단일 소표본으로 망각 제거를 주장하지 않는다.

### 7.2 공식 평가 adapter와 오류 처리

- 설치된 τ의 Agent/LLM interface, message/tool schema, reward basis, grader, 실제 split names를 읽고 연결한다. private evaluator fields가 모델 payload에 섞이지 않는 테스트를 둔다.
- official evaluation criteria를 그대로 이용하고, 일치 여부를 확인하는 known-success/known-failure trajectory fixture를 만든다. grader를 golden 문장 유사도로 대체하지 않는다.
- authentication, CA, URL normalization, timeout, 429/일시 5xx retry, concurrency 제한, request/response 저장을 구현한다. secrets는 redact한다.
- replay key는 task ID, trial, model hash, data revision, config hash다. 실패 episode는 단순 message replay가 아니라 저장 state의 유효성을 확인하거나 초기 상태에서 다시 시작한다.
- 실패·timeout·invalid tool call·budget exhaustion을 denominator에서 제거하지 않는다. attempted/completed/succeeded/failed/unsupported counts를 보고한다.
- official scorer가 호출하는 judge와 user simulator의 비용도 모델 inference 비용과 분리해 기록한다.
- episode wall time, 전체 agent tokens/calls, retrieval/tool latency, 성공률을 보고한다. non-streaming latency로 TTFT를 만들지 않는다.
- task 단위 paired bootstrap CI를 제공하고 반복 trial의 상관을 보존한다. smoke는 5개 episode/1 trial 정도의 경로 확인이며 공식 성능 결론을 내리지 않는다.
- 공식 score와 protocol이 동일하지 않거나 모델이 지원하지 않는 tool 기능을 우회한 경우 결과에 그 차이를 명시한다.

### 7.3 MLflow

Experiments는 `rhoai-model-training-lab-data`, `rhoai-model-training-lab-training`, `rhoai-model-training-lab-evaluation`으로 나눈다. data release run → training run → evaluation suite/variant run을 IDs와 hashes로 연결한다.

| 종류 | 필수 기록 |
|---|---|
| data | KB/τ revision, generator/validator config, 통과·거절·coverage, bundle ID/hash, 검토 상태 |
| train | base/tokenizer, canonical/export hashes, seed, loss, 실제 steps/tokens, duration/VRAM |
| eval tags | track, protocol, variant, execution_mode, verified/not_run/blocked |
| eval config | task/split/revision, grader, user simulator, model artifact, controller, retriever, budgets |
| metrics | 공식 task success/pass^k(가능할 때), diagnostic accuracy, action recall, errors, latency/tokens/calls, retention |
| artifacts | redacted config, manifest, per_task/per_trial JSONL, failures, traces, summary/comparison, plots, scorer identity |

MLflow 기본 평가 저장이 실패하면 성공으로 표시하지 않는다. local 결과를 먼저 보존하고 실패 상태를 남긴다. `--no-mlflow`는 명시적 offline 경로다. `scripts/log_eval_results.py`로 fingerprint 기반 재업로드/중복 방지를 제공한다. GB 단위 weights는 자동 업로드하지 않고 PVC/S3/OCI URI/hash를 기록한다.

```bash
python scripts/run_eval.py --track prepared_diagnostics --config configs/eval.yaml
python scripts/run_eval.py --track tau_episodes --config configs/eval.yaml --limit 5 --trials 1
python scripts/run_eval.py --track retention --config configs/eval.yaml
```

노트북과 CLI는 같은 라이브러리를 호출한다. `08_compare_results.ipynb`는 실제 MLflow 결과를 읽고 task success, 진단 정확도, retention, 지연과 실패 사례를 비교한다. 가상 성능 값은 넣지 않는다.

## 8. 저장소 구조와 구현 계약

| 영역 | 필수 산출물 |
|---|---|
| root | README.md, CLAUDE.md, pyproject.toml, Makefile, .env.example, .gitignore |
| configs | data-preparation.yaml, sdg.yaml, data-release.yaml, lora/osft profiles, rag.yaml, endpoints.example.yaml, eval.yaml |
| environments | preparation, lora, osft, backend, tau-eval의 lock/constraints와 kernel 설치 안내 |
| notebooks | 5.1의 참가자 8개 + 제작자 3개 |
| src/rhoai_model_training_lab | config, schemas, data, sdg, training, deployment, rag, harness, api, tau_adapter, evaluation, tracking |
| scripts | source/SDG/validation/bundle/fetch/verify, train/export/upload/render, index/start/stop/smoke, run_eval/log_eval_results |
| docs | architecture, reference-audit, compatibility, data-card, data-release-guide, evaluation-protocol, troubleshooting, implementation-status |
| deploy | model-serving manifests, backend manifests, Containerfile |
| tests | fixture 데이터, split 누출, backend export parity/loss, tool replay, session isolation, official scorer adapter, resume/MLflow failure |

코드 본체는 src에 두고 노트북마다 복제하지 않는다. .env.example은 teacher/validator/user simulator/base/lora/osft/embedding/MLflow/S3/CA 설정을 분리한다. 참가자 학습 필수값과 데이터 제작/평가 전용값을 표시한다. raw hidden eval, secrets, weights, notebook 실행 출력은 git에 넣지 않는다.

## 9. 완료 조건

| Gate | 증거 |
|---|---|
| G0 조사 | 실제 τ/SDG/Training Hub signatures, versions, tool discovery/reward/split 확인 |
| G1 데이터 | 실제 bundle, hashes, 품질 보고, train/eval 격리, LoRA/OSFT export sample parity |
| G2 준비된 학습 | teacher 없이 bundle로 두 backend loader/loss mask 검증, 가능한 실제 GPU smoke |
| G3 배포 | export reload, manifest, 실제 chat/tool-call endpoint 검증 또는 blocker |
| G4 backend | persistent index, session isolation, planner 분기, 실제 simulator tool 연결, budget 종료 |
| G5 평가 | official success/failure fixture 및 실제 episode, MLflow 저장/조회 또는 blocker |
| G6 재현 | fresh kernel 참가자 경로, 제작자 재생성 경로, 실제 실행 결과와 제한사항 |

`data_ready`, `backend_load_validated`, `gpu_smoke_verified`, `tau_live_verified`를 구분한다. 데이터 파일 생성만으로 GPU 학습 성공을 주장하지 않는다. GPU 없이도 tokenizer/loss/schema validation은 가능한 범위에서 수행한다.

실행 자원이 없으면 코드·절차를 완성하고 해당 gate를 blocked로 남긴다. fixture/dry-run을 live 결과로 포장하지 않는다. 데이터 제작 endpoint가 없으면 dummy를 prepared lab 데이터로 표시하지 않는다. 완료 보고에는 bundle의 실제 준비 상태를 우선 표시한다.

## 10. Opus 공통 실행 프롬프트

```text
너는 RHOAI 3.5, LLM post-training, data engineering, agent evaluation을 구현하는 엔지니어다.
이 rhoai-model-training-lab-opus-spec.md 전체를 읽고 실행 가능한 저장소를 완성하라.
기존 instructions와 사용자 변경을 보존하고 구현 상태를 읽어 이어서 작업하라.

도메인은 τ-Knowledge banking_knowledge로 확정되어 있다.
가장 중요한 요구사항은 DATA-FIRST다:
운영자가 sdg_hub로 데이터를 생성·검증·분할·backend 변환하여 prepared bundle을 먼저 완성하고,
참가자는 teacher API나 SDG 없이 bundle을 받아 간단한 LoRA/OSFT 노트북을 실행한다.

필수 원본:
https://github.com/sierra-research/tau2-bench
https://arxiv.org/abs/2603.04370
https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub
https://github.com/Red-Hat-AI-Innovation-Team/training_hub
https://github.com/cbtham/micro-financial-loan/tree/main
https://github.com/hyogrin/rhoai-custom-research-lab
https://github.com/hyogrin/rhoai-inference-design-planner/blob/main/scripts/run_eval.py

실제 source/signatures와 task/tool/reward/split을 확인하고 version을 고정하라.
upstream main과 제품 이미지 버전을 같다고 가정하지 말라.
G0 → 1A prepared data → 1B training/deployment → 2 harness → 3 evaluation 순으로 구현하라.

반드시 지켜라:
- 실제 sdg_hub, training_hub API를 사용하고 얇은 adapter로 호환성 문제를 격리한다.
- 같은 base와 같은 canonical samples로 LoRA/OSFT를 독립 학습한다.
- 데이터 생성/수정/분할/복잡한 변환을 학습 노트북에서 수행하지 않는다.
- 학습 파일은 두 backend에서 load 가능하고 assistant-only loss가 확인되어야 한다.
- tool trajectories는 schema와 실제 simulator replay로 검증한다.
- 평가 private fields/expected actions/gold docs를 SDG나 agent에 제공하지 않는다.
- 모든 실험군에 같은 업무 tools와 현재 계정 관측 권한을 제공한다.
- τ-derived offline 진단과 공식 multi-turn task success를 분리한다.
- run_eval.py의 MLflow 패턴만 참고하고 100점 문장 유사도를 공식 reward로 쓰지 않는다.
- 모든 모델 평가에는 실제 serving endpoint를 사용한다.
- τ grading 버전, simulator, task/trial, corpus, model/config hash를 MLflow에 저장한다.
- 가상 데이터/점수/URL을 실제 prepared release 또는 성공 결과처럼 만들지 않는다.

산출물은 package, prepared datasets, producer/learner notebooks, scripts,
배포 manifests, backend, τ runner adapter, MLflow evaluation, 의미 있는 tests이다.
설계나 scaffold만 작성하고 종료하지 말라.
코드/fixture 검증은 진행하고 자원 부족은 정확한 blocker와 실행 명령으로 남겨라.
매 단계 docs/implementation-status.md를 갱신하라.
```

## 11. 1A — 사전 데이터 준비 프롬프트

```text
공통 명세를 읽고 먼저 prepared dataset release를 구현하라.
SDG를 참가자 학습 노트북에 넣지 말라.

1. τ의 banking KB, 공개 policy/tool schema, 공식 task/split/hidden fields를 조사하라.
2. 공식 eval task는 보류하고 KB+독립 학습 scenario로 SDG를 설계하라.
   공식 train이 실제로 있으면 사용 범위/오염 검사를 기록하라.
3. sdg_hub YAML flows로 QA, 정책 적용, clarification, tool calls, trajectories를 생성하라.
4. 근거/조건 검증, 독립 validator, 실제 simulator replay, 중복·split 누출 검사를 구현하라.
5. canonical train/validation을 backend별 JSONL로 export하라.
   sample parity, chat template, tool serialization, assistant-only masks, length를 검증하라.
6. manifest/checksums/provenance/data card/quality report/configs를 bundle로 패키징하라.
7. fetch/verify 스크립트, 실제 small smoke bundle과 가능한 lab release를 준비하라.
8. 제작자용 source/SDG/release 노트북 3개를 작성하라.

재배포 가능한 데이터만 포함하고 hidden evaluation bundle은 분리하라.
생성한 조건 추출과 replay에 오류가 없다고 단정하지 말고 검토 범위를 기록하라.
실제 endpoint가 없으면 blocked로 보고하고 dummy 데이터로 G1을 통과시키지 말라.
최종 보고에서 data_ready와 backend_load_validated 상태를 가장 먼저 제시하라.
```

## 12. 1B — 간단한 학습·배포 프롬프트

```text
공통 명세와 prepared bundle manifest를 읽고 참가자 학습 경로를 구현하라.
00 preflight, 01 bundle load, 03 LoRA, 04 OSFT, 05 export/deploy 노트북을 작성하라.
03/04는 설정, 데이터 검사/표본, 학습, 로그, checkpoint 확인 중심의 5~7 실행 셀을 목표로 한다.
원본 수집, teacher 호출, SDG, split, 복잡한 데이터 변환을 넣지 말라.

두 학습은 동일 base revision과 같은 canonical dataset에서 시작한다.
training_hub의 실제 API와 인자를 확인하여 호출하고 주요 동작을 교육적으로 보여라.
데이터 이상 시 자동 수정·재생성이 아니라 validation 오류와 producer 경로를 안내하라.
학습 환경별 lock, GPU preflight, loss masking, checkpoint/resume, MLflow 기록을 구현하라.
LoRA adapter+merged model, OSFT HF export를 저장하고 reload 검증하라.
micro-financial-loan의 export/S3/serving 패턴을 RHOAI 3.5에 맞게 적용하라.
실제 vLLM chat template/tool parser로 tool-call smoke를 수행하라.
GPU 미제공이면 코드·loader 검증과 구체적 자원 blocker를 남기고 성공을 꾸미지 말라.
```

## 13. 2단계 — RAG·공식 simulation 연결 프롬프트

```text
공통 명세와 기존 package를 읽고 RAG/planning backend를 구현하라.
rhoai-custom-research-lab의 graph/state/layers/api/observability를 읽어 작은 구조로 재사용하라.
FastAPI+LangGraph+persistent vector index와 τ simulation adapter를 연결하라.
KB와 train example index를 분리하고 corpus/embedding/version hash를 확인하라.

/v1/qa는 진단용, /v1/agent/step은 공식 agent runner 연결용으로 제공하라.
plan → retrieval/tool discovery → policy reasoning → tool call → observation → verify를 구현하라.
user simulator private instruction, expected actions, final reward state를 agent에게 주지 말라.
업무 tools는 공식 simulation을 연결하고 현재 state를 학습 예제의 값으로 대체하지 말라.
공식 tool discovery를 보존하고 모든 schema 사전 노출을 기본으로 하지 말라.
세션 격리, episode reset, bounded retries/budget, state-changing call 중복 방지를 구현하라.
simple_rag/agent_rag 및 train-example retrieval ablation을 제공하라.
setup/index/start/stop/smoke 스크립트와 06 노트북, Containerfile/manifests를 작성하라.
실제 은행 외부 API는 연결하지 말고 공식 simulation만 사용하라.
```

## 14. 3단계 — 평가·MLflow 프롬프트

```text
공통 명세에 따라 prepared_diagnostics, tau_episodes, retention을 구현하라.
참고 run_eval.py를 읽고 CLI/MLflow 구조만 재사용하라.
FinQA 수치 프로그램 scorer 또는 golden 문장 100점 유사도는 이 저장소의 평가로 사용하지 말라.

τ 공식 runner/grader의 의미를 유지하면서 실제 RHOAI model endpoint와 backend step API를 연결하라.
private task fields가 모델 request에 들어가지 않는지 테스트하라.
동일 controller/업무 도구/simulator에서 Base/LoRA/OSFT × no_knowledge/RAG를 비교하라.
planning과 example retrieval 효과는 별도 통제 실험으로 나누어라.
공식 task success와 pass^k를 사용하고 pass@k로 혼동하지 말라.
단일-turn synthetic 진단은 lab-derived metric으로 구분하라.
known-success/failure trajectories로 grader adapter를 검증하라.
모든 실패·timeout을 denominator에 포함하고 task 단위 paired CI를 지원하라.
07 평가/08 비교 노트북, run_eval/log_eval_results CLI, eval config를 작성하라.
MLflow에 data→train→eval lineage, τ grading revision, simulator/config/model hashes,
공식 reward, 진단, latency/tokens/calls, failures와 redacted traces를 저장하라.
실제 저장/조회와 fixture 검증을 구분하고 offline 재업로드를 지원하라.
```

## 15. 최종 검토 프롬프트

```text
명세 대비 구현을 검토하고 결함을 수정하라.
- teacher API 없이 prepared bundle로 학습 노트북을 실행할 수 있는가?
- 실제 data bytes/manifest/checksum과 품질 보고서가 있는가?
- LoRA/OSFT의 canonical samples와 assistant loss 대상이 일치하는가?
- tool history/serialization과 inference template가 호환되는가?
- synthetic trajectories를 실제 simulation에서 검증했는가?
- KB adaptation과 task leakage를 구분했고 평가 private 필드는 차단했는가?
- 모든 군에 같은 계정 도구/관측 권한이 주어지는가?
- unofficial 진단을 공식 τ 점수로 표시하지 않는가?
- task/trial마다 simulator state가 격리되고 초기화되는가?
- 공식 grading 버전과 pass^k 의미를 보존하는가?
- MLflow lineage/failure/offline 상태와 실제 실행 증거가 일치하는가?
- 참가자 노트북이 fresh kernel에서 짧고 순차적으로 실행 가능한가?
최종 보고는 data readiness, 실제 검증, 미검증/blocked, 실행 명령 순서로 작성하라.
```
