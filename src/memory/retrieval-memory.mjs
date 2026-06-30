/**
 * @file retrieval-memory.mjs
 * @description 基于 Milvus 向量数据库与语义检索的“检索增强记忆（Retrieval Memory / RAG Memory）”机制演示。
 *
 * =================================================================================
 * 核心逻辑与策略解析 (Core Logic & Strategy)
 * =================================================================================
 *
 * 【设计背景与优势】
 * 1. 节省 Token 开销：不需要像传统记忆（如 Buffer Memory）那样每次都把所有历史记录发送给 LLM。
 * 2. 避免注意力分散：如果历史上下文过长或包含大量无关废话，大模型容易发生“Lost in the Middle”（中间信息丢失）的现象，
 *    影响模型的判断和回复质量。检索式记忆通过“按需召回”强语义相关的历史对话片段，确保模型注意力集中。
 * 3. 记忆容量无限：传统的基于滑动窗口或总结的记忆，受限于 LLM 的 Context Window 上限；而检索式记忆将记忆存储在
 *    外部的向量数据库中，物理容量几乎是无限的。无需对数据库中的记录进行总结压缩，而是通过固定召回阈值 K，天然控制上下文长度。
 *
 *
 * =================================================================================
 * 【生产环境避坑指南：记忆连贯性缺陷与混合记忆 (Hybrid Memory)】
 * =================================================================================
 * ⚠️ 警告：纯向量检索记忆在生产环境中存在“记忆不连贯（Lack of Coherence）”的缺陷！
 *
 * 1. 记忆不连贯问题（代词指代丢失）：
 *    - 向量相似度依靠的是词义匹配。如果用户问：“我打算做一个智能农业灌溉项目”，接着问：“你觉得【它】有前景吗？”
 *    - 此时输入只有“你觉得它有前景吗”，语义过于抽象，向量检索很难匹配到上一轮关于“农业灌溉”的具体对话。
 *    - 这会导致代词（如“它”、“那个”、“昨天说的”）指代消解失败，使得对话上下文割裂。
 *
 * 2. 行业标准解法 —— 混合记忆机制 (Hybrid Memory)：
 *    - 短期记忆（Buffer Window）：强制无损将最近的 3~5 轮完整对话拼接在 Prompt 底部，用来维持即时语境和代词连贯性。
 *    - 长期记忆（Vector Retrieval）：使用向量数据库检索更早时间范围内的 Top-K 历史片段，提供长线背景知识。
 *    - 二者拼装作为最终上下文发送给模型，既能流畅聊天，又能唤醒久远记忆。
 * 【流程节点示意 (Workflow)】
 *
 *               用户输入当前问题 (Query)
 *                          │
 *                          ▼
 *          1. 向量化问题 (Embedding Query)
 *                          │
 *                          ▼
 *       2. 检索向量数据库 (Retrieve Top-K similar chats) ◄─── 已经存好的历史对话向量
 *                          │
 *                          ▼
 *          3. 筛选出相似度最高的历史记录 (Top-K Context)
 *                          │
 *                          ▼
 *      4. 组装 Prompt 作为 Context 连同问题一起发给模型
 *                          │
 *                          ▼
 *                 5. 模型生成回答 (LLM Response)
 *                          │
 *                          ▼
 *     6. 将最新对话组装并向量化，存入向量数据库，以供后续轮次检索
 *
 * =================================================================================
 * 关键参数说明
 * =================================================================================
 * - limit / k (默认 2): 检索相关历史对话的最大召回条数。通过固定该值，天然限制了历史对话在上下文中的总 Token 占比。
 * - metric_type: MetricType.COSINE (余弦相似度)，用于衡量向量间的语义接近程度度量。
 */

import "dotenv/config";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { MilvusClient, MetricType } from "@zilliz/milvus2-sdk-node";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const COLLECTION_NAME = "conversations";
const VECTOR_DIM = 1024;

// 初始化 OpenAI Chat 模型
const model = new ChatOpenAI({
  modelName: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0, // 设为 0 以保证生成的对话内容更聚焦和稳定
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

// 初始化 Embeddings 向量模型
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.EMBEDDINGS_API_KEY,
  model: process.env.EMBEDDINGS_MODEL_NAME,
  configuration: {
    baseURL: process.env.EMBEDDINGS_BASE_URL,
  },
  dimensions: VECTOR_DIM,
});

// 初始化 Milvus 客户端连接
const client = new MilvusClient({
  address: "localhost:19530",
});

/**
 * 获取文本的向量嵌入
 *
 * @param {string} text - 需要进行向量化的文本内容
 * @returns {Promise<number[]>} 返回文本的浮点数向量数组
 */
async function getEmbedding(text) {
  const result = await embeddings.embedQuery(text);
  return result;
}

/**
 * 从 Milvus 向量数据库中检索最相关的历史对话
 *
 * @param {string} query - 当前用户的问题
 * @param {number} k - 检索出的相似对话条数，默认返回最相关的 2 条记录
 * @returns {Promise<Array<object>>} 相似的历史对话记录数组
 */
async function retrieveRelevantConversations(query, k = 2) {
  try {
    // 1. 生成查询问题本身的向量嵌入
    const queryVector = await getEmbedding(query);

    // 2. 在 Milvus 向量集合中搜索最相似的对话数据记录
    const searchResult = await client.search({
      collection_name: COLLECTION_NAME,
      vector: queryVector,
      limit: k,
      metric_type: MetricType.COSINE, // 使用余弦相似度衡量语义贴合度
      output_fields: ["id", "content", "round", "timestamp"], // 指定需要返回的字段
    });

    return searchResult.results;
  } catch (error) {
    console.error("检索对话时出错:", error.message);
    return [];
  }
}

/**
 * 策略演示: 检索增强型记忆（Retrieval-Augmented Memory）
 *
 * 使用 Milvus 向量数据库存储每一轮的对话，并根据用户当前的问题，进行语义相似度搜索，
 * 动态地将捞取到的历史对话注入当前的 Prompt 作为背景知识传递给 LLM。
 */
async function retrievalMemoryDemo() {
  try {
    console.log("连接到 Milvus...");
    await client.connectPromise;
    console.log("✓ 已连接\n");
  } catch (error) {
    console.error("❌ 无法连接到 Milvus:", error.message);
    console.log("请确保 Milvus 服务正在运行（localhost:19530）");
    return;
  }

  // 创建一个辅助的本地内存消息历史记录
  const history = new InMemoryChatMessageHistory();

  // 模拟待测试的用户追问（与之前 insert-conversations.mjs 中插入的背景强相关）
  const conversations = [
    { input: "我之前提到的机器学习项目进展如何？" }, // 期待召回 round=2 关于机器学习的对话
    { input: "我周末经常做什么？" }, // 期待召回 round=3、round=4 关于篮球/电影的对话
    { input: "我的职业是什么？" }, // 期待召回 round=1 赵六数据科学家、round=5 软件工程师的对话
  ];

  for (let i = 0; i < conversations.length; i++) {
    const { input } = conversations[i];
    const userMessage = new HumanMessage(input);

    console.log(`\n[第 ${i + 1} 轮对话]`);
    console.log(`用户: ${input}`);

    // 1. 根据当前用户的输入，从 Milvus 向量库中检索语义最相关的历史对话
    console.log("\n【检索相关历史对话】");
    const retrievedConversations = await retrieveRelevantConversations(
      input,
      2, // 限制最大只召回 2 条历史记录
    );

    let relevantHistory = "";
    if (retrievedConversations.length > 0) {
      // 打印展示检索出来的关联对话及其得分
      retrievedConversations.forEach((conv, idx) => {
        console.log(`\n[历史对话 ${idx + 1}] 相似度: ${conv.score.toFixed(4)}`);
        console.log(`轮次: ${conv.round}`);
        console.log(`内容: ${conv.content}`);
      });

      // 组装格式化背景历史数据字符串
      relevantHistory = retrievedConversations
        .map((conv, idx) => {
          return `[历史对话 ${idx + 1}]
轮次: ${conv.round}
${conv.content}`;
        })
        .join("\n\n━━━━━\n\n");
    } else {
      console.log("未找到相关历史对话");
    }

    // 2. 构建 Prompt（如果有检索出来的历史，将其当作背景上下文嵌入到消息中）
    const contextMessages = relevantHistory
      ? [
          new HumanMessage(
            `相关历史对话：\n${relevantHistory}\n\n用户问题: ${input}`,
          ),
        ]
      : [userMessage];

    // 3. 调用模型生成回答
    console.log("\n【AI 回答】");
    const response = await model.invoke(contextMessages);

    // 将本轮的原始交互存入本地的简易历史缓冲中
    await history.addMessage(userMessage);
    await history.addMessage(response);

    // 4. 将当前生成的对话数据，向量化后存入 Milvus 向量数据库中，以供未来的对话轮次进行检索
    const conversationText = `用户: ${input}\n助手: ${response.content}`;
    const convId = `conv_${Date.now()}_${i + 1}`;
    const convVector = await getEmbedding(conversationText);

    try {
      await client.insert({
        collection_name: COLLECTION_NAME,
        data: [
          {
            id: convId,
            vector: convVector,
            content: conversationText,
            round: i + 1,
            timestamp: new Date().toISOString(),
          },
        ],
      });
      console.log(`💾 已保存到 Milvus 向量数据库`);
    } catch (error) {
      console.warn("保存到向量数据库时出错:", error.message);
    }

    console.log(`助手: ${response.content}`);
  }
}

// 启动策略演示
retrievalMemoryDemo().catch(console.error);
