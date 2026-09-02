# 通用知识库 P0 设计

## 目标

将资料库切换为逻辑文档优先模型。物理文件（Asset）不再是检索、版本、权限或评测边界；Document、DocumentVersion、Chunk、Source 与 AccessPolicy 是唯一的跨文件类型语义，允许破坏既有 API、JSONL 与 Qdrant payload。

## 现状与兼容边界

- 新库不读取旧 `asset_index.jsonl` 或旧 `documents.jsonl` 行；首次运行需要重新上传/索引。
- API、CLI、Qdrant collection 和 eval case 统一改用 `document_id`、`version_id`、`chunk_id`；不再输出或接受 `asset_id` 与 `expected_asset_ids`。
- 删除以资产为单位的删除、聚合、缓存和评测匹配路径；资产仍保存物理内容哈希和路径，但不是公共身份。

## 领域模型

- `Source` 表示来源和可选 URL；`Asset` 表示不可变的物理字节、哈希和存储位置。
- `Document` 是稳定公共身份；上传必须生成或提供 `document_id`。
- `DocumentVersion` 用内容哈希标识一版；相同 `document_id + content_hash` 幂等，不同哈希成为同一文档的新版本。
- `Chunk` 是版本内可检索片段，具有稳定 `chunk_id`；`AccessPolicy` 强制 collection、metadata 与允许主体。

解析器直接输出新的 `Chunk` 记录，文档存储使用新 schema。索引逐 chunk 流式写入，避免同时复制整个语料。

## 数据流

`UploadPipeline.confirm` 创建 `DocumentVersion + Asset` 持久记录。解析产生 `Chunk`，Qdrant payload 直接以新字段建立索引并创建 payload indexes。查询必须携带访问上下文，Qdrant 在召回前过滤 collection、metadata、ACL，服务再做防御性校验并按 `document_id` 聚合；响应返回 document/version/chunk 身份与最佳证据。

## 低置信度拒答

最低置信度为强制配置：没有命中或最高分低于阈值时不调用 LLM，返回稳定的“证据不足”响应与空 sources。

## 评测（qrels 过渡）

评测以 qrels 为唯一输入：`{query_id: {document_id: relevance}}`。结果输出 document-level Recall、MRR、MAP 与 graded-NDCG；移除宽松的文件名/标题匹配。

## 验收标准

1. 领域 dataclass 生成稳定 document/version/chunk 身份；同一内容哈希幂等，不同哈希产生递增版本。
2. 所有持久化和 Qdrant payload 只接受新 schema，且有 payload indexes 用于 collection、metadata、ACL 过滤。
3. 搜索只返回授权的 document-level 命中，含 document/version/chunk 身份和最佳证据。
4. 低置信度或空命中拒答不调用 LLM。
5. qrels 按 document_id 评分，输出 graded NDCG、MRR、MAP 与 Recall。

## 回滚

这次切换不提供数据回滚。实施前清空开发数据目录与 Qdrant collections；代码回滚后需重新初始化对应版本的数据。
