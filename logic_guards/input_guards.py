from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Guardrail Definition
class LegalGuardrail:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Define the guardrail prompt and chain
        self.guard_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a security filter for a Canadian Legal Assistant. 
            Determine if the question is related to Canadian law.
            Respond with exactly 'SAFE' or 'UNSAFE'."""),
            ("human", "{query}")
        ])
        
        self.chain = self.guard_prompt | self.llm | StrOutputParser()

    def validate(self, query: str) -> bool:
        response = self.chain.invoke({"query": query}).strip().upper()
        return "SAFE" in response