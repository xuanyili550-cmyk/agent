"""
Gala Agent 用到的工具（不含嘉宾检索，那个在 retriever.py）：
  - DuckDuckGo 搜索
  - 假天气（演示自定义工具）
  - HF Hub 下载统计

三框架共用。同名工具在不同框架里类型不同，故按框架用不同后缀导出：
  smolagents   -> WeatherInfoTool / HubStatsTool（类，调用方自己实例化）+ DuckDuckGoSearchTool
  llama-index  -> search_tool / weather_info_tool_li / hub_stats_tool_li（FunctionTool 实例）
  langgraph    -> DuckDuckGoSearchRun + weather_info_tool_lg / hub_stats_tool_lg（Tool 实例）
本模块只造工具、不联网，import 很轻。
"""

import random
from huggingface_hub import list_models


# 天气/Hub 的纯逻辑函数（三框架复用同一份实现）
def _get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20},
    ]
    data = random.choice(weather_conditions)  # 随机选一种天气
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


def _get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # 列出该作者的模型，按下载量排序
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))
        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"


# =====================================================================
# #smolagents —— 导出 DuckDuckGoSearchTool（转发）+ 两个 Tool 子类
# =====================================================================
from smolagents import DuckDuckGoSearchTool, Tool as _SmolTool


class WeatherInfoTool(_SmolTool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for.",
        }
    }
    output_type = "string"

    def forward(self, location: str):
        return _get_weather_info(location)


class HubStatsTool(_SmolTool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from.",
        }
    }
    output_type = "string"

    def forward(self, author: str):
        return _get_hub_stats(author)


# =====================================================================
# #llam-index —— FunctionTool 实例（后缀 _li 避免与其它框架重名）
# =====================================================================
from llama_index.core.tools import FunctionTool
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

search_tool = FunctionTool.from_defaults(DuckDuckGoSearchToolSpec().duckduckgo_full_search)
weather_info_tool_li = FunctionTool.from_defaults(_get_weather_info)
hub_stats_tool_li = FunctionTool.from_defaults(_get_hub_stats)


# =====================================================================
# #langgraph —— 转发 DuckDuckGoSearchRun + langchain Tool 实例（后缀 _lg）
# =====================================================================
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool as _LcTool

weather_info_tool_lg = _LcTool(
    name="get_weather_info",
    func=_get_weather_info,
    description="Fetches dummy weather information for a given location.",
)
hub_stats_tool_lg = _LcTool(
    name="get_hub_stats",
    func=_get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub.",
)
