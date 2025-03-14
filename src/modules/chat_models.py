from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


class ChatModel:
    
    def __init__(
        self, 
        provider: str, 
        model_name: str, 
        temperature: float|int, 
        max_tokens: int, 
        role: str = None
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.role = role
        
    def fetch_model(self):
        
        engine = None
        
        if self.provider == 'openai':
            engine = ChatOpenAI
        elif self.provider == 'anthropic':
            engine = ChatAnthropic
        else:
            raise NotImplementedError
        
        return engine(
            model = self.model_name,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
        )