#integrating the agents with the playground
from phi.agent import Agent
import phi.api
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq

import os
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

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

app=Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)
