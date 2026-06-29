import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import {
  HumanMessage,
  AIMessage,
  trimMessages,
} from "@langchain/core/messages";
import { getEncoding } from "js-tiktoken";

// ===== 1.消息数量的来截断 =====
async function messageCountTruncation() {
  const history = new InMemoryChatMessageHistory();
  const maxMessages = 4;

  const messages = [
    { type: "human", content: "我叫张三" },
    { type: "ai", content: "你好张三，很高兴认识你！" },
    { type: "human", content: "我今年25岁" },
    { type: "ai", content: "25岁正是青春年华，有什么我可以帮助你的吗？" },
    { type: "human", content: "我喜欢编程" },
    { type: "ai", content: "编程很有趣！你主要用什么语言？" },
    { type: "human", content: "我住在北京" },
    { type: "ai", content: "北京是个很棒的城市！" },
    { type: "human", content: "我的职业是软件工程师" },
    { type: "ai", content: "软件工程师是个很有前景的职业！" },
  ];

  // 添加所有消息
  for (const msg of messages) {
    if (msg.type === "human") {
      await history.addMessage(new HumanMessage(msg.content));
    } else {
      await history.addMessage(new AIMessage(msg.content));
    }
  }

  let allMessages = await history.getMessages();

  // 按消息数量截断：保留最近 maxMessages 条消息
  const trimmedMessages = allMessages.slice(-maxMessages);

  console.log(`保留消息数量: ${trimmedMessages.length}`);
  console.log(
    "保留的消息:",
    trimmedMessages
      .map((m) => `${m.constructor.name}: ${m.content}`)
      .join("\n  "),
  );
}

// 计算消息数组的总 token 数量
function countTokens(messages, encoder) {
  let total = 0;

  for (const msg of messages) {
    const content =
      typeof msg.content === "string"
        ? msg.content
        : JSON.stringify(msg.content);
    total += encoder.encode(content).length;
  }
  console.log(`total ${total}`);
  return total;
}

// ==== 2. 按 token 数量截断（使用 js-tiktoken 计数 ====
async function tokenCountTruncation() {
  console.log("[tokenCountTruncation start] \n");
  const history = new InMemoryChatMessageHistory();
  const maxTokens = 100; // 限制最多 100 个 token
  const enc = getEncoding("cl100k_base");

  const messages = [
    { type: "human", content: "我叫李四" },
    { type: "ai", content: "你好李四，很高兴认识你！" },
    { type: "human", content: "我是一名设计师" },
    {
      type: "ai",
      content: "设计师是个很有创造力的职业！你主要做什么类型的设计？",
    },
    { type: "human", content: "我喜欢艺术和音乐" },
    { type: "ai", content: "艺术和音乐都是很好的爱好，它们能激发创作灵感。" },
    { type: "human", content: "我擅长 UI/UX 设计" },
    { type: "ai", content: "UI/UX 设计非常重要，好的用户体验能让产品更成功！" },
  ];

  // 添加所有消息
  for (const msg of messages) {
    if (msg.type === "human") {
      await history.addMessage(new HumanMessage(msg.content));
    } else {
      await history.addMessage(new AIMessage(msg.content));
    }
  }

  let allMessages = await history.getMessages();

  // 使用 trimMessages API：使用 js-tiktoken 计算 token 数量
  /**
   * 
   * trimMessages 每次调用countTokens会去逐步递减 -1条的方式试探 allMessages 的条数，直到满足 maxTokens.
   * total 136  <- 第一次：包含全部 8 条消息（超标 100）
    total 130  <- 第二次：丢弃最早的第 1 条，计算剩余 7 条（超标 100）
    total 114  <- 第三次：再丢弃第 2 条，计算剩余 6 条（超标 100）
    total 107  <- 第四次：再丢弃第 3 条，计算剩余 5 条（超标 100）
    total 78   <- 第五次：再丢弃第 4 条，计算剩余 4 条（78 <= 100，安全！）
    total 78   <- 确定最终结果，返回这 4 条消息
   * 
   */
  const trimmedMessages = await trimMessages(allMessages, {
    maxTokens: maxTokens,
    tokenCounter: async (msgs) => countTokens(msgs, enc),
    strategy: "last", // 保留最近的消息最晚的开始
  });

  // 计算实际 token 数用于显示
  const totalTokens = countTokens(trimmedMessages, enc);

  console.log(`总 token 数: ${totalTokens}/${maxTokens}`);
  console.log(`保留消息数量: ${trimmedMessages.length}`);
  console.log(
    "保留的消息:",
    trimmedMessages
      .map((m) => {
        const content =
          typeof m.content === "string" ? m.content : JSON.stringify(m.content);
        const tokens = enc.encode(content).length;
        return `${m.constructor.name} (${tokens} tokens): ${content}`;
      })
      .join("\n  "),
  );
}
async function runAll() {
  await messageCountTruncation();
  await tokenCountTruncation();
}

runAll().catch(console.error);
