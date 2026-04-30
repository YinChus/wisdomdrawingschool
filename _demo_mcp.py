"""验收脚本：演示 ToolRouter 的多厂商能力 + 透明调用"""
import asyncio
from mcp_client import ToolRouter


async def main():
    r = ToolRouter()
    print("=== 5 家厂商可直接用的 schema ===\n")
    print("OpenAI / DeepSeek / Qwen / GLM / Moonshot 通用：", len(r.openai_schema()), "个工具")
    print("Anthropic Claude 原生：", len(r.anthropic_schema()), "个工具")
    print("Google Gemini：", len(r.gemini_schema()[0]["functionDeclarations"]), "个工具\n")

    print("=== 实际调用（mode=" + r.mode + "，透明走 inproc/stdio/sse 均可）===")
    cat = await r.call_tool("classify_math", content="椭圆 x^2/4+y^2=1 的离心率")
    print("  classify_math →", cat)
    style = await r.call_tool("get_style", category=cat)
    print("  get_style →", style)
    prompt = await r.call_tool(
        "generate_geogebra_prompt",
        category=cat,
        content="椭圆 x^2/4+y^2=1 的离心率",
        solution="e=√3/2",
    )
    print("  generate_geogebra_prompt →", prompt[:120].replace("\n", " "), "...")
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
