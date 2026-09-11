# LangGraph 与 LangChain 有何不同？
# LangChain 提供了一个标准接口，用于与模型和其他组件交互，这对于数据检索、LLM 调用和工具调用非常有用。LangChain 中的类可以在 LangGraph 中使用，但并非必须使用。
#
# 这两个软件包各不相同，可以单独使用，但最终，你在网上找到的所有资源都是将这两个软件包结合使用的。

# LangGraph 在以下关键场景中表现出色：
#
# 需要对流程进行明确控制的多步骤推理过程
# 需要步骤间保持状态的应用
# 将确定性逻辑与人工智能能力相结合的系统
# 需要人为干预的工作流程
# 具有多个组件协同工作的复杂代理架构
# LangGraph 的工作原理是什么？
# 它的核心LangGraph是使用有向图结构来定义应用程序的流程：
# 节点代表各个处理步骤（例如调用 LLM、使用工具或做出决定）。
# 边缘定义了步骤之间可能的过渡。
# 状态由用户定义和维护，并在执行过程中于节点之间传递。在决定下一个目标节点时，我们查看的就是当前状态。
from typing_extensions import TypedDict
#状态是 LangGraph 的核心概念
class State(TypedDict):
    graph_state: str

#节点
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}

# LLM 调用：生成文本或做出决策
# 工具调用：与外部系统交互
# 条件逻辑：确定下一步
# 人工干预：获取用户反馈
#边连接节点，并定义图中可能的路径
import random
from typing import Literal
def decide_mood(state) -> Literal["node_2", "node_3"]:
    # 通常，我们会使用状态来决定要访问的下一个节点
    user_input = state['graph_state']

    # 我们就简单地在节点 2 和 3 之间进行 50/50 的分配
    if random.random() < 0.5:
        # 50% 的情况下，我们返回节点 2，
        return "node_2"

   # 50% 的情况下，我们返回节点 3，
    return "node_3"
#状态图是承载整个代理工作流程的容器
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
# 构建图表
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
# 逻辑
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)
# 添加图
graph = builder.compile()
# 视图
display(Image(graph.get_graph().draw_mermaid_png()))
graph.invoke({"graph_state" : "Hi, this is Lance."})

#构建你的第一个语言图
import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

#定义一下Alfred在电子邮件处理工作流程中需要跟踪哪些信息
class EmailState(TypedDict):
    # 正在处理的电子邮件
    email: Dict[str, Any]  # 包含主题、发件人、正文等。

    # 邮件类别（咨询、投诉等）
    email_category: Optional[str]

    # 邮件被标记为垃圾邮件的原因
    spam_reason: Optional[str]

    # 分析和决策
    is_spam: Optional[bool]

    # 回复生成
    email_draft: Optional[str]

    # 处理元数据
    messages: List[Dict[str, Any]]   # 跟踪与 LLM 的对话以进行分析

#步骤二：定义节点
# 初始化我们的 LLM
model = ChatOpenAI(temperature=0)


def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]

    # 这里我们可以进行一些初始预处理
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")

    # 此处无需状态更改，
    return {}


def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]

    # 准备 LLM
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """

    # 致电 LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    # 用于解析响应的简单逻辑（在实际应用中，您需要更强大的解析功能）
    response_text = response.content.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text

    # 如果是垃圾邮件，则提取原因
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()

    # 确定邮件类别是否合法
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break

    # 更新用于跟踪的消息
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    # 返回状态更新
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }


def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")

    # 我们已完成处理此邮件，
    return {}


def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"

    # 准备 LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    This email has been categorized as: {category}

    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    # 致电 LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    # 更新用于跟踪的消息
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    # 返回状态更新
    return {
        "email_draft": response.content,
        "messages": new_messages
    }


def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]

    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-" * 50)
    print(state["email_draft"])
    print("=" * 50 + "\n")

    # 我们已完成处理此邮件，
    return {}

#步骤 3：定义路由逻辑
def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"

#步骤 4：创建状态图并定义边
# 创建图表
email_graph = StateGraph(EmailState)

# 添加节点
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

# 开始添加边
email_graph.add_edge(START, "read_email")
# 添加边 - 定义流
email_graph.add_edge("read_email", "classify_email")

# 添加来自 classify_email 的条件分支
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_response"
    }
)

# 添加最终边
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

# 编译图
compiled_graph = email_graph.compile()

#步骤 5：运行应用程序
# 合法电子邮件示例
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
}

# 垃圾邮件示例
spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
}

# 处理合法电子邮件
print("\nProcessing legitimate email...")
legitimate_result = compiled_graph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})

# 处理垃圾邮件
print("\nProcessing spam email...")
spam_result = compiled_graph.invoke({
    "email": spam_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})


#第六步：使用 Langfuse 检查我们的邮件分拣代理

import os

# 从项目设置页面获取项目密钥：https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region
from langfuse.langchain import CallbackHandler

# 初始化 LangGraph/Langchain 的 Langfuse 回调处理器（跟踪）
langfuse_handler = CallbackHandler()

# 处理合法电子邮件
legitimate_result = compiled_graph.invoke(
    input={"email": legitimate_email, "is_spam": None, "spam_reason": None, "email_category": None, "draft_response": None, "messages": []},
    config={"callbacks": [langfuse_handler]}
)

#可视化我们的图表
compiled_graph.get_graph().draw_mermaid_png()

#文档分析图
# 该系统可以：
# 流程图像文档
# 使用视觉模型（视觉语言模型）提取文本
# 必要时进行计算（以演示常用工具）
# 分析内容并提供简明扼要的摘要
# 执行与文件相关的具体指令
import base64
from typing import List, TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
#定义智能体的状态
class AgentState(TypedDict):
    # 提供的文档
    input_file: Optional[str]  # 包含文件路径（PDF/PNG）
    messages: Annotated[list[AnyMessage], add_messages]


vision_llm = ChatOpenAI(model="gpt-4o")
def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.

    Master Wayne often leaves notes with his training regimen or meal plans.
    This allows me to properly analyze the contents.
    """
    all_text = ""
    try:
        # 读取图像并以 base64 编码
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 准备包含 base64 图像数据的提示信息
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # 呼叫具备视觉功能的模型
        response = vision_llm.invoke(message)

        # 追加提取的文本
        all_text += response.content + "\n\n"

        return all_text.strip()
    except Exception as e:
        # 管家应该优雅地处理错误
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""


def divide(a: int, b: int) -> float:
    """Divide a and b - for Master Wayne's occasional calculations."""
    return a / b


#给管家配备工具
tools = [
    divide,
    extract_text
]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

#节点
def assistant(state: AgentState):
    # 系统消息
    textual_description_of_tool="""
extract_text(img_path: str) -> str:
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (strings).

    Returns:
        A single string containing the concatenated text extracted from each image.
divide(a: int, b: int) -> float:
    Divide a and b
"""
    image=state["input_file"]
    sys_msg = SystemMessage(content=f"You are a helpful butler named Alfred that serves Mr. Wayne and Batman. You can analyse documents and run computations with provided tools:\n{textual_description_of_tool} \n You have access to some optional images. Currently the loaded image is: {image}")

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }

#ReAct模式
#图表
builder = StateGraph(AgentState)

# 定义节点：这些节点负责执行构建工作
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 定义边：这些边决定控制流的移动方式
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # 如果最新消息需要工具，则路由至工具页面
    # 否则，提供直接回复
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# 显示管家的思考过程
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

#管家在行动例1：简单计算
messages = [HumanMessage(content="Divide 6790 by 5")]
messages = react_graph.invoke({"messages": messages, "input_file": None})

# 显示messages
for m in messages['messages']:
    m.pretty_print()

#分析
messages = [HumanMessage(content="According to the note provided by Mr. Wayne in the provided images. What's the list of items I should buy for the dinner menu?")]
messages = react_graph.invoke({"messages": messages, "input_file": "Batman_training_and_meals.png"})
