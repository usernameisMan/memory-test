/**
 * @file summarization-memory-by-token.mjs
 * @description 基于 Token 限制的滑动窗口“总结式记忆（Summarization Memory）”机制演示。
 *
 * =================================================================================
 * 核心流程解析 (Flow Analysis)
 * =================================================================================
 *
 * 【设计背景】
 * 在构建 LLM 对话应用时，上下文长度受限且输入 Token 费用昂贵。此机制提供了一种“滑动压缩”历史对话的策略：
 * 既保留最新的对话细节（无损上下文），又通过 LLM 总结老旧的对话（有损压缩），防止上下文溢出。
 *
 * 【流程节点示意】
 *     开始: 接收并存入新消息
 *       │
 *       ▼
 *     获取全部对话历史
 *       │
 *       ▼
 *     计算所有消息的总 Token 数 (totalTokens)
 *       │
 *       ├─► [未超阈值] ──► 保持现状，继续对话
 *       │
 *       └─► [已超阈值] ──► 从最新消息反向累加 Token
 *                            │
 *                            ▼
 *                         锁定保留的最新消息 (recentMessages)
 *                            │
 *                            ▼
 *                         截取较早的消息进行总结 (messagesToSummarize)
 *                            │
 *                            ▼
 *                         调用大模型生成旧对话的总结 (summary)
 *                            │
 *                            ▼
 *                         清空历史并重载 recentMessages
 *                            │
 *                            ▼
 *                         结束压缩，准备下一次对话
 *
 * =================================================================================
 * 核心逻辑与策略说明
 * =================================================================================
 * - 触发条件：当历史总 Token 数量超过 200 时，触发总结流程。
 * - 滑动保留策略：从最新消息（数组尾部）开始反向逐条累加。如果一条消息加进来后总共不超过 80 个 Token，
 *   它就会被完整且无损地保留；一旦某条消息加入后会导致累计值超过 80，该遍历便会熔断中断。
 * 
 * =================================================================================
 * 关键参数说明
 * =================================================================================
 * - maxTokens (默认 200): 触发总结的最大 Token 阈值。
 * - keepRecentTokens (默认 80): 最新消息保留额度。
 *
 * =================================================================================
 * 滑动窗口切片算法
 * =================================================================================
 * 1. 从对话历史的最新一条消息（数组尾部）开始向前遍历。
 * 2. 累加最新消息的 Token 数量。只要累加值不超过 keepRecentTokens (80)，就放入 `recentMessages` 队列中（使用 unshift 保证原有先后顺序）。
 * 3. 一旦累加值超过 80 Token，立即中断循环。
 * 4. 剩余的较早对话则被切片为 `messagesToSummarize` 范围，由大模型进行 Summarize 归纳总结。
 * 5. 清空原历史，重新写入 `recentMessages`。在实际生产中，总结内容（summary）可存于 SystemMessage 或外部存储中，作为上下文拼接到后续对话输入。
 */

import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  getBufferString,
} from "@langchain/core/messages";
import { getEncoding } from "js-tiktoken";

// 初始化 LangChain ChatOpenAI 客户端实例
// 参数从环境变量加载，包括模型名称、API Key 和自定义的 Base URL
const model = new ChatOpenAI({
  modelName: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0, // 设为 0 以保证生成的对话总结更加确定和稳定
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

/**
 * 计算消息数组的总 token 数量
 *
 * @param {Array<import("@langchain/core/messages").BaseMessage>} messages - 消息对象数组
 * @param {import("js-tiktoken").Tiktoken} encoder - tiktoken 编码器实例
 * @returns {number} 所有消息内容的总 token 数量
 */
function countTokens(messages, encoder) {
  let total = 0;
  for (const msg of messages) {
    // 确保消息内容是字符串类型，如果不是，则进行 JSON 序列化
    const content =
      typeof msg.content === "string"
        ? msg.content
        : JSON.stringify(msg.content);
    // 计算并累加该条消息的 token 数量
    total += encoder.encode(content).length;
  }
  return total;
}

// ========== 总结策略演示（基于 token 计数） ==========
/**
 * 演示基于 Token 限制的“总结式记忆（Summarization Memory）”的核心流程。
 *
 * 主要步骤：
 * 1. 初始化内存对话历史 `InMemoryChatMessageHistory` 和 Tiktoken 编码器。
 * 2. 模拟多轮关于“红烧肉做法”的对话并存入历史。
 * 3. 检查当前所有消息的总 Token 数是否超出了设定的阈值 `maxTokens` (200)。
 * 4. 如果未超出：保留完整历史，不做任何处理。
 * 5. 如果超出：
 *    a. 从后向前遍历历史消息，保留尽量多的“最近消息”传入 `recentMessages`，直到总长度接近 `keepRecentTokens` (80)。
 *    b. 剩余的较早消息（超出保留范围的）被划分为“待总结消息” `messagesToSummarize`。
 *    c. 调用大模型对“待总结消息”进行总结。
 *    d. 清空原对话历史，把“最近消息”重新存入，完成一轮“记忆滑动与压缩”。
 */
async function summarizationMemoryDemo() {
  const history = new InMemoryChatMessageHistory();
  const maxTokens = 200; // 触发总结的最大 Token 阈值
  const keepRecentTokens = 80; // 滑动窗口中需要被无损保留的“最新消息” Token 额度

  // 初始化 cl100k_base 编码器（常用于 gpt-3.5-turbo 和 gpt-4）
  const enc = getEncoding("cl100k_base");

  // 模拟的多轮复杂对话数据
  const messages = [
    { type: "human", content: "我想学做红烧肉，你能教我吗？" },
    {
      type: "ai",
      content:
        "当然可以！红烧肉是一道经典的中式菜肴。首先需要准备五花肉、冰糖、生抽、老抽、料酒等材料。",
    },
    { type: "human", content: "五花肉需要切多大块？" },
    {
      type: "ai",
      content:
        "建议切成3-4厘米见方的块，这样既容易入味，口感也更好。切好后可以用开水焯一下去除血沫。",
    },
    { type: "human", content: "炒糖色的时候有什么技巧吗？" },
    {
      type: "ai",
      content:
        "炒糖色是关键步骤。用小火慢慢炒，等冰糖完全融化变成焦糖色，冒小泡时就可以下肉了。注意不要炒过头，否则会发苦。",
    },
    { type: "human", content: "需要炖多长时间？" },
    {
      type: "ai",
      content:
        "一般需要炖40-60分钟，用小火慢炖，直到肉变得软糯入味。可以用筷子戳一下，能轻松戳透就说明好了。",
    },
    { type: "human", content: "最后收汁的时候要注意什么？" },
    {
      type: "ai",
      content:
        "收汁时要用大火，不断翻动，让汤汁均匀包裹在肉块上。看到汤汁变得浓稠，颜色红亮就可以出锅了。",
    },
  ];

  // 1. 将模拟的对话消息依次转换为 LangChain 的消息对象并存入历史记录中
  for (const msg of messages) {
    if (msg.type === "human") {
      await history.addMessage(new HumanMessage(msg.content));
    } else {
      await history.addMessage(new AIMessage(msg.content));
    }
  }

  // 2. 获取目前为止所有的历史消息
  let allMessages = await history.getMessages();

  // 3. 计算所有消息的总 Token 消耗量
  const totalTokens = countTokens(allMessages, enc);

  // 4. 检查是否达到了需要总结压缩的阈值
  if (totalTokens >= maxTokens) {
    const recentMessages = [];
    let recentTokens = 0;

    // 5. 核心滑动窗口算法：从最新的消息开始（即数组尾部）向后遍历，
    //    尽可能多地保留最新消息，只要它们的 Token 累加值不超过 `keepRecentTokens`。
    for (let i = allMessages.length - 1; i >= 0; i--) {
      const msg = allMessages[i];
      const content =
        typeof msg.content === "string"
          ? msg.content
          : JSON.stringify(msg.content);
      const msgTokens = enc.encode(content).length;

      // 如果加上当前消息的总 Token 仍然在保留额度内，则将其保存在“保留消息”中
      if (recentTokens + msgTokens <= keepRecentTokens) {
        recentMessages.unshift(msg); // 保持原有的先后顺序，推入数组头部
        recentTokens += msgTokens;
      } else {
        // 一旦超过了额度，剩余的更早的消息将不再被保留，直接中断循环
        break;
      }
    }

    // 6. 确定需要总结的旧消息范围：从数组开头到保留消息开始之前的这一段
    const messagesToSummarize = allMessages.slice(
      0,
      allMessages.length - recentMessages.length,
    );
    const summarizeTokens = countTokens(messagesToSummarize, enc);

    console.log("\n💡 Token 数量超过阈值，开始总结...");
    console.log(
      `📝 将被总结的消息数量: ${messagesToSummarize.length} (${summarizeTokens} tokens)`,
    );
    console.log(
      `📝 将被保留的消息数量: ${recentMessages.length} (${recentTokens} tokens)`,
    );

    // 7. 调用大模型对需要丢弃的旧对话进行归纳总结
    const summary = await summarizeHistory(messagesToSummarize);

    // 8. 清空内存中的旧历史，释放空间，并把保留下来的“最新消息”重新写入历史记录中
    //    在实际生产中，我们可以将 `summary` 存储在 SystemMessage 或是外部数据库中，
    //    以便在下一轮对话中作为上下文（Context）传递给大模型。
    await history.clear();
    for (const msg of recentMessages) {
      await history.addMessage(msg);
    }

    console.log(`\n保留消息数量: ${recentMessages.length}`);
    console.log(
      "保留的消息:",
      recentMessages
        .map((m) => {
          const content =
            typeof m.content === "string"
              ? m.content
              : JSON.stringify(m.content);
          const tokens = enc.encode(content).length;
          return `${m.constructor.name} (${tokens} tokens): ${m.content}`;
        })
        .join("\n  "),
    );
    console.log(`\n总结内容（不包含保留的消息）: ${summary}`);
  } else {
    // 9. 如果未达到阈值，则输出提示并继续保持当前的完整上下文
    console.log(
      `\nToken 数量 (${totalTokens}) 未超过阈值 (${maxTokens})，无需总结`,
    );
  }
}

// 执行演示函数并捕获可能出现的异常
summarizationMemoryDemo().catch(console.error);

/**
 * 总结历史对话的函数
 *
 * @param {Array<import("@langchain/core/messages").BaseMessage>} messages - 需要被总结的消息对象数组
 * @returns {Promise<string>} 返回总结后的字符串内容
 */
async function summarizeHistory(messages) {
  // 如果没有需要总结的消息，直接返回空字符串
  if (messages.length === 0) return "";

  // 将消息历史转换为可读的文本段落（如：用户: xxx \n 助手: yyy）
  const conversationText = getBufferString(messages, {
    humanPrefix: "用户",
    aiPrefix: "助手",
  });

  // 构建专门的总结提示词（Prompt）
  const summaryPrompt = `请总结以下对话的核心内容，保留重要信息：

${conversationText}

总结：`;

  // 调用模型进行总结，并将结果返回
  const summaryResponse = await model.invoke([
    new SystemMessage(summaryPrompt),
  ]);
  return summaryResponse.content;
}
