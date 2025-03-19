from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo # for websearch

#web search agent
web_search_agent = Agent(
    name='web_search_agent',
    role = "Search for the Information",
    model=Groq( id ="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always include the source of the information in the answer"],
    show_tools_calls=True,
    markdown=True
)

#finance agent
finance_agent = Agent(
    name='finance_agent',
    role = "Finance Information",
    model=Groq( id ="llama3-70b-8192"),
    tools=[
        YFinanceTools(stock_price = True,analyst_recommendations=True,stock_fundamentals=True,company_news=True)
        ],
    instructions=["Use tables to display the information"],
    show_tools_calls=True,
    markdown=True
)

multi_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama3-70b-8192"),
    instructions=["Use the finance agent to get the stock price and then use the web search agent to find the latest news about stock","Always include the source of the information in the answer","Use tables to display the information"],
    show_tools_calls=True,
    markdown=True
)

multi_agent.print_response("Summarize analyst recommendations and share the latest news for COCHINSHIP", stream=True)