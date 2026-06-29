import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { FileSystemChatMessageHistory } from "@langchain/community/stores/message/file_system";
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
} from "@langchain/core/messages";

import path from "node:path";

const model = new ChatOpenAI({
  modelName: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

async function fileHistoryDemo() {
  //指定储存文件的路径
  const filePath = path.join(process.cwd(), "chat_history.json");
  //指定对话id
  const sessionId = "user_session_001";

  // 系统提示词
  const systemMassage = new SystemMessage(
    "你是一个友好的、幽默的做菜助手,喜欢分享美食和烹饪技巧",
  );

  const restoredhistory = new FileSystemChatMessageHistory({
    filePath,
    sessionId,
  });

  const restoreMessages = await restoredhistory.getMessages();

  console.log(`本地文件恢复了 ${restoreMessages.length} 条对话信息`);

  restoreMessages.forEach((msg, index) => {
    const type = msg.type;

    const prefix = type === "human" ? "用户" : "助手";

    console.log(`对话 ${index + 1}. [${prefix}]: ${msg.content}`);
  });

  console.log("[第三轮对话]");
  const userMassage3 = new HumanMessage("需要那些食材?");

  await restoredhistory.addMessage(userMassage3);

  const message3 = [systemMassage, ...(await restoredhistory.getMessages())];
  const response3 = await model.invoke(message3);
  await restoredhistory.addMessage(response3);

  console.log(`用户：${userMassage3.content}`);
  console.log(`助手：${response3.content}`);
  console.log(`对话已保存到文件：${filePath} \n`);
}

fileHistoryDemo();
