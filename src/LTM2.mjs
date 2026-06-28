import 'dotenv/config'
import { ChatOpenAI } from '@langchain/openai'
import { FileSystemChatMessageHistory } from "@langchain/community/stores/message/file_system"
import {
    HumanMessage,
    AIMessage,
    SystemMessage,
} from "@langchain/core/messages"