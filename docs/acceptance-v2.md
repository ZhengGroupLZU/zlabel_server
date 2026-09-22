# v2 验收清单（M5）

逐条对应 `docs/architecture-v2.md` §14 的 DoD。每条给出**自动化证据**（仓库内可复跑）与**手工步骤**（真实 OpenList/GPU 环境）。
跑自动化：

```bash
cd zlabel_server && uv run pytest -q            # 220 passed, 3 failed(仅 CUDA 环境)
cd .. && uv run pytest -q                        # 557 passed, 3 skipped
cd zlabel_server && uv run pytest tests/v2/test_client_contract.py -q   # 12 项：客户端 ↔ 真实服务端
```

## 1. 两人抢同一帧
- 自动化：`tests/v2/test_tasks.py::test_claim_then_conflict_for_the_second_annotator`、
  `tests/v2/test_client_contract.py::test_lease_expiry_and_takeover`、
  `tests/gui/test_workflow.py::test_claim_conflict_locks_the_frame`
- 手工：两个客户端（一个 reviewer/admin、一个 annotator）打开同一帧 → 后开者弹出
  "该帧已被领取：<名字> / 租约到期 …"，三个按钮（只读查看 / 下一帧 / 接管）可用；该帧保持只读，
  Finish/Submit 禁用；把 `ZLSERVER_LEASE_MINUTES=1` 后等待租约过期 → 后者可直接领取。

## 2. 推理只依赖本次请求的帧
- 自动化：`tests/v2/test_inference_worker.py::test_same_frame_is_encoded_once`、
  `::test_different_frames_are_encoded_separately`、`::test_results_follow_the_requested_frame`、
  `tests/v2/test_predict.py::test_each_predict_sends_its_own_frame`
- 手工：同一帧上连续点 3 次（第 2、3 次明显更快，`/metrics` 命中率上升）；交替预测两张不同的图，
  掩膜分别对应各自的图。

## 3. 角色与审计
- 自动化：`tests/v2/test_auth.py::test_user_admin_endpoints_are_admin_only`、
  `tests/v2/test_tasks.py::test_review_requires_a_reviewer_and_a_note_on_reject`、
  `::test_my_stats_and_audit_trail`
- 手工：annotator 账号看不到 Fetch 扫描/复核/标签管理入口（按钮禁用或提示需要 reviewer）；
  越权请求返回 403；`audit_log` 表里应有 `claim/save_annotation/submit/review_approve/...` 记录，
  `user_id` 为操作者。

## 4. 版本冲突
- 自动化：`tests/v2/test_annotations.py::test_stale_base_version_conflicts`、
  `::test_force_overwrite_is_reviewer_only`、
  `tests/v2/test_client_contract.py::test_annotation_roundtrip_and_optimistic_locking`
- 手工：客户端 A 保存后，客户端 B 用旧版本保存 → 弹"标注冲突"，消息含服务端版本/作者/时间；
  "强制覆盖"仅 reviewer 可见且生效。

## 5. 进度与统计
- 自动化：`tests/v2/test_tasks.py::test_progress_reflects_task_states`、
  `tests/v2/test_projects.py::test_progress_by_user`、
  `tests/v2/test_client_contract.py::test_claim_lease_and_review_roundtrip`（`my-stats`）
- 手工：提交/复核几帧后，状态栏 `Done: approved/total` 与
  `GET /api/v2/projects/<p>/progress?by_user=true` 一致。

## 6. 推理进程故障隔离
- 自动化：`tests/v2/test_predict.py::test_worker_unavailable_is_503`、
  `tests/v2/test_inference_worker.py::test_queue_full_is_503`
- 手工：停掉 worker → 点击预测提示"推理不可用，可继续手动标注"，标注/保存/复核/取图全部照常；
  重新启动 worker（API 不用重启）→ 预测恢复。

## 7. 双端版本匹配
- 自动化：`tests/v2/test_health.py::test_v1_era_clients_are_told_to_upgrade`（旧客户端得到 410 +
  升级说明）、`tests/v2/test_client_contract.py::test_login_carries_user_role_and_capabilities`
- 手工：用旧版客户端连新服务器 → 明确提示升级；用本版客户端连旧服务器 → 登录时报版本不匹配。

## 8. 数据面与迁移
- 自动化：`tests/v2/test_migrations.py`（`upgrade head` / `alembic check` 无漂移 / `downgrade base`）
- 手工：全新库 `uv run alembic upgrade head` → 起服务 → 登录即用；历史 `.zlabel` 文件可直接读取
  （`anno_id = md5("<project>/<相对路径>")` 未变）。

## 已知不在 v2 范围
- 旧库（`zlabel_server.db`）的数据不迁移（决策 D5）；`/api/v1` 已移除（返回 410）。
- 剩余 3 个失败用例恒为 `tests/inference/test_cuda.py`：本机无可用 CUDA/cuBLAS，GPU 机器上应通过
  （`uv run pytest -m gpu`）。
