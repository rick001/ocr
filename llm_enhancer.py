import os
import openai
from dotenv import load_dotenv
import json
import re
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

class LLMEnhancer:
    """
    LLM-based text enhancement using DeepSeek model via OpenRouter.ai
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM enhancer.
        
        Args:
            api_key: OpenRouter API key. If None, will try to get from OPENROUTER_API_KEY env var
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        print(f"Debug: API key found: {'Yes' if self.api_key else 'No'}")
        print(f"Debug: API key length: {len(self.api_key) if self.api_key else 0}")
        
        # Configure OpenAI client for OpenRouter with version-agnostic approach
        try:
            print("Debug: Initializing OpenAI client...")
            
            # Try to get the OpenAI version to understand the issue
            import openai
            print(f"Debug: OpenAI version: {openai.__version__}")
            
            # Use a more basic approach that should work across versions
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            print("Debug: OpenAI client initialized successfully")
            
        except TypeError as e:
            print(f"Debug: TypeError during initialization: {str(e)}")
            if "proxies" in str(e):
                print("Debug: Proxies error detected, trying version-specific workaround...")
                try:
                    # Try using the client with minimal parameters and manual base_url setting
                    self.client = openai.OpenAI(api_key=self.api_key)
                    
                    # Set the base URL manually after initialization
                    if hasattr(self.client, 'base_url'):
                        self.client.base_url = "https://openrouter.ai/api/v1"
                    else:
                        # If base_url attribute doesn't exist, try setting it differently
                        print("Debug: No base_url attribute, trying alternative approach...")
                        # Try to set it through the client's internal configuration
                        if hasattr(self.client, '_client'):
                            # Some versions store the base URL in the internal client
                            pass
                    
                    print("Debug: Version-specific workaround successful")
                    
                except Exception as e2:
                    print(f"Debug: Version-specific workaround failed: {str(e2)}")
                    
                    # Last resort: Try creating a custom client configuration
                    try:
                        print("Debug: Trying custom client configuration...")
                        
                        # Import httpx directly to avoid the wrapper issue
                        import httpx
                        
                        # Create a custom httpx client without proxies
                        http_client = httpx.Client(
                            timeout=httpx.Timeout(30.0),
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            }
                        )
                        
                        # Create OpenAI client with custom http_client
                        self.client = openai.OpenAI(
                            api_key=self.api_key,
                            base_url="https://openrouter.ai/api/v1",
                            http_client=http_client
                        )
                        
                        print("Debug: Custom client configuration successful")
                        
                    except Exception as e3:
                        print(f"Debug: Custom client configuration failed: {str(e3)}")
                        raise ValueError(f"Failed to initialize OpenAI client after all attempts. Last error: {str(e3)}")
            else:
                raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
        except Exception as e:
            print(f"Debug: Unexpected error during initialization: {str(e)}")
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
        
        # DeepSeek model identifier
        self.model = "deepseek/deepseek-chat-v3-0324:free"
        print("Debug: LLM Enhancer initialized successfully")
    
    def enhance_ocr_text(self, raw_text: str, context: str = "document") -> str:
        """
        Enhance OCR text using LLM for better interpretation and formatting.
        
        Args:
            raw_text: Raw text extracted from OCR
            context: Context of the document (e.g., "document", "receipt", "form")
            
        Returns:
            Enhanced and corrected text
        """
        if not raw_text.strip():
            return raw_text
        
        try:
            # Create prompt for text enhancement
            prompt = self._create_enhancement_prompt(raw_text, context)
            
            # Call LLM for enhancement
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at interpreting and correcting OCR text. Your task is to enhance raw OCR output by fixing errors, improving formatting, and making the text more readable while preserving the original meaning and structure."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=4000
            )
            
            enhanced_text = response.choices[0].message.content.strip()
            return enhanced_text
            
        except Exception as e:
            print(f"LLM enhancement failed: {str(e)}")
            # Return original text if LLM fails
            return raw_text
    
    def _create_enhancement_prompt(self, raw_text: str, context: str) -> str:
        """
        Create a prompt for text enhancement.
        
        Args:
            raw_text: Raw OCR text
            context: Document context
            
        Returns:
            Formatted prompt for LLM
        """
        return f"""
Please enhance and correct the following OCR text from a {context}. 

Raw OCR text:
{raw_text}

Please:
1. Fix spelling and grammar errors
2. Correct OCR misinterpretations (e.g., "0" vs "O", "1" vs "l")
3. Improve formatting and readability
4. Preserve the original structure and meaning
5. Maintain proper paragraph breaks
6. Fix common OCR issues like broken words, missing spaces, etc.

Return only the enhanced text without any explanations or markdown formatting.
"""
    
    def extract_structured_data(self, text: str, data_type: str = "general") -> Dict[str, Any]:
        """
        Extract structured data from text using LLM.
        
        Args:
            text: Text to analyze
            data_type: Type of data to extract ("receipt", "invoice", "form", "general")
            
        Returns:
            Dictionary with extracted structured data
        """
        try:
            prompt = self._create_extraction_prompt(text, data_type)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured data from documents. Extract relevant information and return it as valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                # Remove any markdown formatting
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                return json.loads(result_text.strip())
            except json.JSONDecodeError:
                # If JSON parsing fails, return a simple structure
                return {
                    "extracted_text": text,
                    "confidence": "low",
                    "error": "Failed to parse structured data"
                }
                
        except Exception as e:
            print(f"Structured data extraction failed: {str(e)}")
            return {
                "extracted_text": text,
                "confidence": "low",
                "error": str(e)
            }
    
    def _create_extraction_prompt(self, text: str, data_type: str) -> str:
        """
        Create a prompt for structured data extraction.
        
        Args:
            text: Text to analyze
            data_type: Type of data to extract
            
        Returns:
            Formatted prompt for LLM
        """
        if data_type == "receipt":
            return f"""
Extract structured data from this receipt text. Return as JSON with these fields:
- vendor_name: Store or business name
- total_amount: Total amount paid
- date: Date of transaction
- items: Array of items purchased with prices
- tax_amount: Tax amount if mentioned
- payment_method: How payment was made

Text: {text}

Return only valid JSON without any explanations.
"""
        elif data_type == "invoice":
            return f"""
Extract structured data from this invoice text. Return as JSON with these fields:
- invoice_number: Invoice number
- vendor_name: Company name
- client_name: Customer name
- total_amount: Total amount
- date: Invoice date
- due_date: Due date if mentioned
- line_items: Array of items/services with descriptions and amounts

Text: {text}

Return only valid JSON without any explanations.
"""
        else:
            return f"""
Extract key information from this {data_type} text. Return as JSON with relevant fields based on the content.

Text: {text}

Return only valid JSON without any explanations.
"""
    
    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """
        Create a summary of the text using LLM.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized text
        """
        try:
            prompt = f"""
Summarize the following text in {max_length} characters or less:

{text}

Provide a concise summary that captures the main points and key information.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating concise summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Text summarization failed: {str(e)}")
            # Return a simple truncation if LLM fails
            return text[:max_length] + "..." if len(text) > max_length else text 