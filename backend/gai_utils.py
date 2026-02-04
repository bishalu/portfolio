from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import EmbeddingsClient
import google.generativeai as genai
from typing import Union, List, Literal, Type, Dict
import enum
from openai import OpenAI, AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel
import json
import numpy as np
import logging
import asyncio

# Configure logging to suppress Azure library verbosity
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure.ai.inference').setLevel(logging.WARNING)

try:
    from utils.aws_utils import get_secret
except:
    from aws_utils import get_secret


DEFAULT_OAI_MODEL = 'gpt-4.1'
DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash'
DEFAULT_PERPLEXITY_MODEL = 'sonar'


class gai_client:
    """Unified LLM client supporting Azure OpenAI and Gemini with thinking"""
    
    def __init__(self):
        self._initialize_providers()
    
    @staticmethod
    def create_enum_from_options(name: str, options: List[str]) -> Type[enum.Enum]:
        """
        Helper: Create an enum class from a list of string options
        
        Args:
            name: Name for the enum class  
            options: List of string values
            
        Returns:
            Enum class
            
        Example:
            Priority = gai_client.create_enum_from_options("Priority", ["low", "medium", "high"])
        """
        enum_dict = {
            option.upper().replace(' ', '_').replace('-', '_'): option 
            for option in options
        }
        return enum.Enum(name, enum_dict)
    
    @staticmethod
    def is_gemini_compatible_schema(model_class: Type[BaseModel]) -> tuple[bool, List[str]]:
        """
        Helper: Check if a Pydantic model is compatible with Gemini API
        
        Args:
            model_class: Pydantic BaseModel class
            
        Returns:
            Tuple of (is_compatible, list_of_issues)
            
        Example:
            compatible, issues = gai_client.is_gemini_compatible_schema(MyModel)
        """
        issues = []
        annotations = getattr(model_class, '__annotations__', {})
        
        for field_name, field_type in annotations.items():
            type_str = str(field_type)
            
            # Check for Dict types
            if 'Dict[' in type_str or type_str.startswith('dict'):
                issues.append(f"Field '{field_name}' uses Dict type - use enum/Literal instead")
            
            # Check for Any types  
            if 'Any' in type_str:
                issues.append(f"Field '{field_name}' uses Any type - specify concrete types")
        
        return len(issues) == 0, issues

    def _initialize_providers(self):
        """Initialize both Azure OpenAI and Gemini clients"""
        # Gemini initialization
        try:
            gemini_secret = get_secret('vibeset/gemini')
            if gemini_secret and 'Google Gemini API' in gemini_secret:
                genai.configure(api_key=gemini_secret['Google Gemini API'])
                self.gemini_client = genai
                print("✅ Gemini client initialized successfully")
            else:
                print("⚠️ Gemini API key not found - Gemini functionality disabled")
                self.gemini_client = None
        except Exception as e:
            print(f"⚠️ Gemini initialization failed: {e}")
            self.gemini_client = None

    def openai_text(self,
                    prompt: str,
                    model: str = DEFAULT_OAI_MODEL,
                    temperature: float = 0.1,
                    max_tokens: int = 4095,
                    sysmsg: str = None,
                    to_json: bool = False,
                    json_schema: dict = None,
                    collect_tokens: bool = False,
                    strict: bool = False) -> Union[str, dict]:
        """Azure OpenAI text generation with optional token collection"""

        # Validate model
        allowed_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini", "gpt-5-nano", "mistral-small-2503", "Ministral-3B"]
        if model not in allowed_models:
            raise ValueError(f"Invalid model. Supported: {allowed_models}")

        # Build messages
        messages = []
        if sysmsg:
            messages.append({"role": "system", "content": sysmsg})
        messages.append({"role": "user", "content": prompt})

        # Configure response format
        response_format = None
        if to_json:
            if json_schema:
                response_format = json_schema
            else:
                response_format = "json_object"  # String format for Azure inference SDK

        # Get client configuration for model
        client_config = self._get_azure_client_config(model)

        # GPT-5 models use the OpenAI SDK (AzureOpenAI) instead of ChatCompletionsClient
        if model in ['gpt-5-mini', 'gpt-5-nano']:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_version=client_config['api_version'],
                azure_endpoint=client_config['endpoint'].replace(f'/openai/deployments/{model}', ''),
                api_key=client_config['api_key']
            )

            # Build completion params for OpenAI SDK
            # Note: GPT-5 models only support temperature=1.0 (fixed reasoning temperature)
            # Note: GPT-5 uses ~500+ tokens for internal reasoning before output
            completion_params = {
                'model': model,
                'messages': messages,
                'max_completion_tokens': max_tokens,
                'temperature': 1.0,  # GPT-5 models require fixed temperature
            }

            if to_json:
                if json_schema:
                    # Convert Azure JsonSchemaFormat to OpenAI json_schema format
                    completion_params['response_format'] = self._convert_to_openai_schema(json_schema, strict=strict)
                else:
                    # Loose JSON mode
                    completion_params['response_format'] = {"type": "json_object"}

            completion = client.chat.completions.create(**completion_params)
            response_text = completion.choices[0].message.content

        # Mistral Small and Ministral-3B use standard OpenAI SDK with Azure MaaS endpoint
        elif model in ['mistral-small-2503', 'Ministral-3B']:
            client = OpenAI(
                base_url=client_config['endpoint'],
                api_key=client_config['api_key']
            )

            # Build completion params for OpenAI SDK
            completion_params = {
                'model': model,
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
            }

            if to_json:
                # Mistral supports json_object response format
                completion_params['response_format'] = {"type": "json_object"}
                
                # Ensure system message mentions JSON (MANDATORY for json_object mode)
                has_json_instruction = any("json" in m.get("content", "").lower() for m in messages if m.get("role") == "system")
                if not has_json_instruction:
                     if messages and messages[0]['role'] == 'system':
                         messages[0]['content'] += " You must output valid JSON."
                     else:
                         messages.insert(0, {"role": "system", "content": "You are a helpful assistant. You must output valid JSON."})
                     # Update messages in params
                     completion_params['messages'] = messages

            completion = client.chat.completions.create(**completion_params)
            response_text = completion.choices[0].message.content
            if response_text is None:
                 error_msg = f"Mistral content is None! Finish reason: {completion.choices[0].finish_reason}"
                 logging.error(error_msg)
                 raise ValueError(error_msg)

        else:
            # Existing models use ChatCompletionsClient
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential

            client = ChatCompletionsClient(
                endpoint=client_config['endpoint'],
                credential=AzureKeyCredential(client_config['api_key']),
                api_version=client_config['api_version']
            )

            # Handle model-specific parameters
            completion_params = {
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature if model != 'o4-mini' else 1.0,
                'top_p': 1.0,
                'model': model,
                'response_format': response_format
            }

            # Make API call
            completion = client.complete(**completion_params)
            response_text = completion.choices[0].message.content

        # Parse JSON if requested
        if to_json:
            try:
                import json
                response_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text[:200]}")
        else:
            response_data = response_text

        # Return with or without token information
        if collect_tokens:
            # Extract token usage from completion
            usage = getattr(completion, 'usage', None)
            if usage:
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
            else:
                # Fallback estimation if usage not available
                input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
                output_tokens = len(response_text.split()) * 1.3
                input_tokens, output_tokens = int(input_tokens), int(output_tokens)

            # Calculate cost
            cost = PricingRegistry.calculate_cost('azure', model, input_tokens, output_tokens)

            return {
                'response': response_data,
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens
                },
                'cost': cost,
                'model': model,
                'provider': 'azure'
            }

        return response_data

    async def async_openai_text(self,
                                prompt: str,
                                model: str = DEFAULT_OAI_MODEL,
                                temperature: float = 0.1,
                                max_tokens: int = 4095,
                                sysmsg: str = None,
                                to_json: bool = False,
                                json_schema: dict = None,
                                collect_tokens: bool = False,
                                strict: bool = False) -> Union[str, dict]:
        """
        Async version of openai_text for true parallel execution.

        Use this method when you need multiple LLM calls to run concurrently.
        All parameters and return values are identical to openai_text().

        Example:
            results = await asyncio.gather(
                client.async_openai_text("prompt 1", model="gpt-4.1-mini"),
                client.async_openai_text("prompt 2", model="gpt-4.1-nano"),
                client.async_openai_text("prompt 3", model="gpt-4.1-mini")
            )
        """
        # Validate model
        allowed_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini", "gpt-5-nano", "mistral-small-2503", "Ministral-3B"]
        if model not in allowed_models:
            raise ValueError(f"Invalid model. Supported: {allowed_models}")

        # Build messages
        messages = []
        if sysmsg:
            messages.append({"role": "system", "content": sysmsg})
        messages.append({"role": "user", "content": prompt})

        # Get client configuration
        client_config = self._get_azure_client_config(model)

        # Create async client
        if model in ['mistral-small-2503', 'Ministral-3B']:
            client = AsyncOpenAI(
                base_url=client_config['endpoint'],
                api_key=client_config['api_key']
            )
        else:
            client = AsyncAzureOpenAI(
                api_version=client_config['api_version'],
                azure_endpoint=client_config['endpoint'].replace(f'/openai/deployments/{model}', ''),
                api_key=client_config['api_key']
            )
        
        # Mistral Small and Ministral-3B use standard AsyncOpenAI SDK
        if model in ['mistral-small-2503', 'Ministral-3B']:
            client = AsyncOpenAI(
                base_url=client_config['endpoint'],
                api_key=client_config['api_key']
            )

        # Build completion params
        completion_params = {
            'model': model,
            'messages': messages,
            'temperature': 1.0 if model in ['gpt-5-mini', 'gpt-5-nano', 'o4-mini'] else temperature,
        }
        
        if model in ['mistral-small-2503', 'Ministral-3B']:
            completion_params['max_tokens'] = max_tokens
        else:
            completion_params['max_completion_tokens'] = max_tokens

        # Configure response format
        if to_json:
            # Unified logic for OAI and Mistral (since Mistral on Azure supports json_schema now)
            if json_schema:
                 completion_params['response_format'] = self._convert_to_openai_schema(json_schema, strict=strict)
            else:
                 # Fallback to json_object mode
                 completion_params['response_format'] = {"type": "json_object"}
                 
                 # Only strictly enforce JSON instruction if NOT using json_schema (json_schema implies it)
                 # But sticking to safety: Ensure "json" is in system prompt just in case
                 if model in ['mistral-small-2503', 'Ministral-3B']:
                     has_json_instruction = any("json" in m.get("content", "").lower() for m in messages if m.get("role") == "system")
                     if not has_json_instruction:
                          if messages and messages[0]['role'] == 'system':
                              messages[0]['content'] += " You must output valid JSON."
                          else:
                              messages.insert(0, {"role": "system", "content": "You are a helpful assistant. You must output valid JSON."})
                          completion_params['messages'] = messages

        # Make async API call
        completion = await client.chat.completions.create(**completion_params)
        response_text = completion.choices[0].message.content

        # Parse JSON if requested
        if to_json:
            try:
                import json
                response_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text[:200]}")
        else:
            response_data = response_text

        # Return with or without token information
        if collect_tokens:
            usage = getattr(completion, 'usage', None)
            if usage:
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
            else:
                # Fallback estimation
                input_tokens = len(prompt.split()) * 1.3
                output_tokens = len(response_text.split()) * 1.3
                input_tokens, output_tokens = int(input_tokens), int(output_tokens)

            cost = PricingRegistry.calculate_cost('azure', model, input_tokens, output_tokens)

            return {
                'response': response_data,
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens
                },
                'cost': cost,
                'model': model,
                'provider': 'azure'
            }

        return response_data

    def _convert_to_openai_schema(self, azure_schema, strict: bool = False) -> dict:
        """
        Convert Azure JsonSchemaFormat to OpenAI json_schema format with optional strict mode

        Azure format: JsonSchemaFormat(schema={...}, name="SchemaName")
        OpenAI format: {
            "type": "json_schema",
            "json_schema": {
                "name": "schema_name",
                "strict": True/False,
                "schema": {...}  # Must have additionalProperties: false if strict=True
            }
        }

        Args:
            azure_schema: Azure JsonSchemaFormat or dict
            strict: Enable strict mode (requires all fields in 'required', no optional fields)
                   Default False for GPT-5 models to support optional fields
        """
        # Extract schema and name from Azure JsonSchemaFormat
        if hasattr(azure_schema, 'schema'):
            # It's a JsonSchemaFormat object
            schema_dict = azure_schema.schema.copy()
            schema_name = getattr(azure_schema, 'name', 'response_schema')
        else:
            # Already a dict (fallback)
            schema_dict = azure_schema.copy() if isinstance(azure_schema, dict) else azure_schema
            schema_name = 'response_schema'

        # Only add additionalProperties: false if strict mode is enabled
        if strict and 'additionalProperties' not in schema_dict:
            schema_dict['additionalProperties'] = False

        # Convert to OpenAI format
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name.lower().replace(' ', '_'),
                "strict": strict,
                "schema": schema_dict
            }
        }

    def _get_azure_client_config(self, model: str) -> dict:
        """Get Azure client configuration for specific model"""
        secret = get_secret('vibeset/azure_ai_foundry')

        if model in ['o4-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-5-mini', 'gpt-5-nano']:
            return {
                'endpoint': f"https://kevin-m86fxp36-eastus2.cognitiveservices.azure.com/openai/deployments/{model}",
                'api_key': secret.get('api_key_o3'),
                'api_version': '2024-12-01-preview' if model in ['gpt-5-mini', 'gpt-5-nano'] else ('2025-02-01-preview' if model != 'o4-mini' else '2025-01-01-preview')
            }
        elif model == 'mistral-small-2503':
            return {
                'endpoint': "https://kevin-m86fxp36-eastus2.services.ai.azure.com/openai/v1/",
                'api_key': secret.get('api_key_o3'),
                'api_version': None # Not used for OpenAI client with base_url
            }
        elif model == 'Ministral-3B':
            return {
                'endpoint': "https://kevin-m86fxp36-eastus2.services.ai.azure.com/openai/v1/",
                'api_key': secret.get('api_key_o3'),
                'api_version': None # Not used for OpenAI client with base_url
            }

        else:  # gpt-4o, gpt-4o-mini
            return {
                'endpoint': f"https://vibesetbackend9912493372.cognitiveservices.azure.com/openai/deployments/{model}",
                'api_key': secret.get('api_key'),
                'api_version': '2025-02-01-preview'
            }

    
    def gemini_text(self,
                   prompt: str,
                   model: str = DEFAULT_GEMINI_MODEL,
                   temperature: float = 0.3,
                   max_tokens: int = 4095,
                   sysmsg: str = None,
                   to_json: bool = False,
                   json_schema: BaseModel = None,
                   thinking_level: str = "simple",
                   collect_tokens: bool = False
) -> Union[str, dict]:
        """
        Gemini text generation with thinking support
        
        Parameters:
        - thinking_level: "simple" (0), "medium" (512), "complex" (2048)
        - json_schema: Direct Pydantic BaseModel (supports enums and Literal types)
        - All other parameters identical to openai_text()
        
        Note: For flexible schemas, use enum.Enum or typing.Literal instead of Dict[str, Any]
        """
        
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized - check API key configuration")
        
        # Validate model - support all three models with thinking
        allowed_models = [
            "gemini-2.5-pro",        # State-of-the-art thinking model
            "gemini-2.5-flash",      # Best price-performance with thinking
            "gemini-2.5-flash-lite"  # Cost-optimized with thinking
        ]
        if model not in allowed_models:
            raise ValueError(f"Invalid Gemini model. Supported: {allowed_models}")
        
        # Configure thinking via temperature and other parameters (no special ThinkingConfig needed)
        thinking_adjustments = {
            "simple": {"temp_boost": 0.0, "tokens_boost": 0},
            "medium": {"temp_boost": 0.1, "tokens_boost": 500},  
            "complex": {"temp_boost": 0.2, "tokens_boost": 1000}
        }
        adjustment = thinking_adjustments.get(thinking_level, thinking_adjustments["simple"])
        

        # Base configuration with thinking adjustments
        config = {
            "temperature": min(1.0, temperature + adjustment["temp_boost"]),
            "max_output_tokens": max_tokens + adjustment["tokens_boost"]
        }
        
        # Configure JSON output with native Pydantic support
        if json_schema:
            # Structured JSON with Pydantic schema converted to minimal Gemini format
            # Gemini only supports basic JSON Schema subset, not full spec
            if hasattr(json_schema, 'model_json_schema'):
                full_schema = json_schema.model_json_schema()
                # Convert to minimal Gemini-compatible format
                schema_dict = self._convert_to_gemini_schema(full_schema)
            else:
                schema_dict = json_schema  # Fallback if already a dict
            config.update({
                "response_mime_type": "application/json",
                "response_schema": schema_dict
            })
        elif to_json:
            # Basic JSON mode without schema enforcement
            config.update({
                "response_mime_type": "application/json"
            })
        # Plain text mode: no additional config
        
        # Build prompt
        full_prompt = f"{sysmsg}\n\n{prompt}" if sysmsg else prompt
        
        try:
            # Configure safety settings - use proper Gemini format and try "OFF" workaround
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Get the model instance
            model = self.gemini_client.GenerativeModel(model)
            response = model.generate_content(
                contents=full_prompt,
                generation_config=config,
                safety_settings=safety_settings
            )
            
            # Handle blocked responses more gracefully
            response_text = None
            if response.candidates:
                # Try to get text from the first candidate
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        response_text = candidate.content.parts[0].text
                
                # If content is blocked, check safety ratings and log
                if not response_text and hasattr(candidate, 'safety_ratings'):
                    blocked_reasons = []
                    for rating in candidate.safety_ratings:
                        if rating.blocked:
                            blocked_reasons.append(f"{rating.category.name}: {rating.probability.name}")
                    
                    if blocked_reasons:
                        logging.warning(f"Gemini response blocked due to: {', '.join(blocked_reasons)}")
                        # Return a fallback response instead of failing
                        response_text = '{"error": "Response blocked by safety filters", "blocked_reasons": "' + ', '.join(blocked_reasons) + '"}'
            
            # Fallback if no candidates or content
            if not response_text:
                logging.warning("Gemini returned no candidates or content")
                response_text = '{"error": "No response generated"}'
            
            # Parse JSON if schema provided or to_json requested (same as openai_text)
            response_data = response_text
            if json_schema or to_json:
                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text[:200]}")

            # Return with or without token information
            if collect_tokens:
                # Extract token usage from response
                usage_metadata = getattr(response, 'usage_metadata', None)
                if usage_metadata:
                    input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                    output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
                else:
                    # Fallback estimation if usage not available
                    input_tokens = len(full_prompt.split()) * 1.3  # Rough token estimation
                    output_tokens = len(response_text.split()) * 1.3
                    input_tokens, output_tokens = int(input_tokens), int(output_tokens)

                # Calculate cost
                cost = PricingRegistry.calculate_cost('gemini', model, input_tokens, output_tokens)

                return {
                    'response': response_data,
                    'usage': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens
                    },
                    'cost': cost,
                    'model': model,
                    'provider': 'gemini'
                }

            return response_data
            
        except Exception as e:
            logging.error(f"Gemini generation failed: {e}")
            raise RuntimeError(f"Gemini API error: {str(e)}")

    def perplexity_text(self,
                       prompt: str,
                       model: str = DEFAULT_PERPLEXITY_MODEL,
                       temperature: float = 0.1,
                       max_tokens: int = 4095,
                       sysmsg: str = None,
                       to_json: bool = False,
                       json_schema: dict = None,
                       collect_tokens: bool = False,
                       search_domain_filter: list = None,
                       search_recency_filter: str = None,
                       return_images: bool = False,
                       return_related_questions: bool = False,
                       search_mode: str = "web") -> Union[str, dict]:
        """
        Perplexity Sonar API text generation with real-time web search.
        Uses OpenAI client library with Perplexity base URL for compatibility.

        Parameters:
        - prompt: Input prompt
        - model: Perplexity model ("sonar", "sonar-pro", "sonar-reasoning", etc.)
        - temperature: Sampling temperature (0-2)
        - max_tokens: Maximum output tokens
        - sysmsg: System message
        - to_json: Return JSON format
        - json_schema: JSON schema for structured output
        - collect_tokens: Collect token usage and cost data
        - search_domain_filter: List of domains to include/exclude
        - search_recency_filter: Filter by content recency ("hour", "day", "week", "month", "year")
        - return_images: Include image URLs in response
        - return_related_questions: Include related questions
        - search_mode: "web" (default) or "academic"

        Returns:
        - String response or dict with response/search results if collect_tokens=True
        """

        # Validate model
        allowed_models = [
            "sonar",                    # Lightweight search model
            "sonar-pro",               # Advanced search model
            "sonar-reasoning",         # Fast reasoning with search
            "sonar-reasoning-pro",     # Precise reasoning with CoT
            "sonar-deep-research"      # Expert-level research model
        ]
        if model not in allowed_models:
            raise ValueError(f"Invalid Perplexity model. Supported: {allowed_models}")

        try:
            # Get Perplexity API key from AWS Secrets Manager
            perplexity_secret = get_secret('vibeset/perplexity')
            api_key = perplexity_secret['Perplexity API Key']

            # Initialize OpenAI client with Perplexity base URL
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )

            # Build messages
            messages = []
            if sysmsg:
                messages.append({"role": "system", "content": sysmsg})
            messages.append({"role": "user", "content": prompt})

            # Build completion parameters
            completion_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Add Perplexity-specific parameters via extra_body
            extra_body = {}
            if search_domain_filter:
                extra_body["search_domain_filter"] = search_domain_filter
            if search_recency_filter:
                extra_body["search_recency_filter"] = search_recency_filter
            if return_images:
                extra_body["return_images"] = return_images
            if return_related_questions:
                extra_body["return_related_questions"] = return_related_questions
            if search_mode != "web":
                extra_body["search_mode"] = search_mode

            if extra_body:
                completion_params["extra_body"] = extra_body

            # Configure JSON mode if requested
            if to_json:
                if json_schema:
                    # Use structured JSON with schema - leverage existing conversion logic
                    # Convert dict schema to pseudo-Azure format for consistency
                    if isinstance(json_schema, dict):
                        # Create a pseudo-Azure schema object for conversion
                        class PseudoAzureSchema:
                            def __init__(self, schema_dict):
                                self.schema = schema_dict
                                self.name = "response_schema"

                        pseudo_schema = PseudoAzureSchema(json_schema)
                        completion_params["response_format"] = self._convert_to_openai_schema(pseudo_schema, strict=False)
                    else:
                        # Direct usage if already in correct format
                        completion_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "response_schema",
                                "strict": False,
                                "schema": json_schema
                            }
                        }
                else:
                    # Basic JSON mode
                    completion_params["response_format"] = {"type": "json_object"}

            # Make API call
            completion = client.chat.completions.create(**completion_params)
            response_text = completion.choices[0].message.content

            # Parse JSON if requested
            if to_json:
                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text[:200]}")
            else:
                response_data = response_text

            # Return with or without token information and search results
            if collect_tokens:
                # Extract token usage
                usage = getattr(completion, 'usage', None)
                if usage:
                    input_tokens = getattr(usage, 'prompt_tokens', 0)
                    output_tokens = getattr(usage, 'completion_tokens', 0)
                else:
                    # Fallback estimation
                    input_tokens = len(prompt.split()) * 1.3
                    output_tokens = len(response_text.split()) * 1.3
                    input_tokens, output_tokens = int(input_tokens), int(output_tokens)

                # Calculate cost (Perplexity has hybrid pricing - tokens + requests)
                cost = PricingRegistry.calculate_cost('perplexity', model, input_tokens, output_tokens, request_count=1)

                # Extract search results if available
                search_results = getattr(completion, 'search_results', [])
                citations = getattr(completion, 'citations', [])

                return {
                    'response': response_data,
                    'usage': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens
                    },
                    'cost': cost,
                    'model': model,
                    'provider': 'perplexity',
                    'search_results': search_results,
                    'citations': citations
                }

            return response_data

        except Exception as e:
            logging.error(f"Perplexity generation failed: {e}")
            raise RuntimeError(f"Perplexity API error: {str(e)}")

    async def async_perplexity_text(self,
                                   prompt: str,
                                   model: str = DEFAULT_PERPLEXITY_MODEL,
                                   temperature: float = 0.1,
                                   max_tokens: int = 4095,
                                   sysmsg: str = None,
                                   to_json: bool = False,
                                   json_schema: dict = None,
                                   collect_tokens: bool = False,
                                   search_domain_filter: list = None,
                                   search_recency_filter: str = None,
                                   return_images: bool = False,
                                   return_related_questions: bool = False,
                                   search_mode: str = "web") -> Union[str, dict]:
        """
        Async version of perplexity_text for true parallel execution.
        All parameters and return values are identical to perplexity_text().
        """

        # Validate model
        allowed_models = [
            "sonar", "sonar-pro", "sonar-reasoning",
            "sonar-reasoning-pro", "sonar-deep-research"
        ]
        if model not in allowed_models:
            raise ValueError(f"Invalid Perplexity model. Supported: {allowed_models}")

        try:
            # Get Perplexity API key from AWS Secrets Manager
            perplexity_secret = get_secret('vibeset/perplexity')
            api_key = perplexity_secret['Perplexity API Key']

            # Initialize async OpenAI client with Perplexity base URL
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )

            # Build messages
            messages = []
            if sysmsg:
                messages.append({"role": "system", "content": sysmsg})
            messages.append({"role": "user", "content": prompt})

            # Build completion parameters
            completion_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Add Perplexity-specific parameters via extra_body
            extra_body = {}
            if search_domain_filter:
                extra_body["search_domain_filter"] = search_domain_filter
            if search_recency_filter:
                extra_body["search_recency_filter"] = search_recency_filter
            if return_images:
                extra_body["return_images"] = return_images
            if return_related_questions:
                extra_body["return_related_questions"] = return_related_questions
            if search_mode != "web":
                extra_body["search_mode"] = search_mode

            if extra_body:
                completion_params["extra_body"] = extra_body

            # Configure JSON mode if requested
            if to_json:
                if json_schema:
                    # Use structured JSON with schema - leverage existing conversion logic
                    if isinstance(json_schema, dict):
                        # Create a pseudo-Azure schema object for conversion
                        class PseudoAzureSchema:
                            def __init__(self, schema_dict):
                                self.schema = schema_dict
                                self.name = "response_schema"

                        pseudo_schema = PseudoAzureSchema(json_schema)
                        completion_params["response_format"] = self._convert_to_openai_schema(pseudo_schema, strict=False)
                    else:
                        # Direct usage if already in correct format
                        completion_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "response_schema",
                                "strict": False,
                                "schema": json_schema
                            }
                        }
                else:
                    completion_params["response_format"] = {"type": "json_object"}

            # Make async API call
            completion = await client.chat.completions.create(**completion_params)
            response_text = completion.choices[0].message.content

            # Parse JSON if requested
            if to_json:
                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text[:200]}")
            else:
                response_data = response_text

            # Return with or without token information and search results
            if collect_tokens:
                # Extract token usage
                usage = getattr(completion, 'usage', None)
                if usage:
                    input_tokens = getattr(usage, 'prompt_tokens', 0)
                    output_tokens = getattr(usage, 'completion_tokens', 0)
                else:
                    # Fallback estimation
                    input_tokens = len(prompt.split()) * 1.3
                    output_tokens = len(response_text.split()) * 1.3
                    input_tokens, output_tokens = int(input_tokens), int(output_tokens)

                # Calculate cost using correct provider detection
                provider = PricingRegistry.detect_provider_for_model(model)
                cost = PricingRegistry.calculate_cost(provider, model, input_tokens, output_tokens, request_count=1)

                # Extract search results if available
                search_results = getattr(completion, 'search_results', [])
                citations = getattr(completion, 'citations', [])

                return {
                    'response': response_data,
                    'usage': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens
                    },
                    'cost': cost,
                    'model': model,
                    'provider': 'perplexity',
                    'search_results': search_results,
                    'citations': citations
                }

            return response_data

        except Exception as e:
            logging.error(f"Async Perplexity generation failed: {e}")
            raise RuntimeError(f"Async Perplexity API error: {str(e)}")

    def _convert_to_gemini_schema(self, full_schema: dict) -> dict:
        """
        IMPROVED: Convert full JSON schema to Gemini-compatible format.

        Preserves:
        - items for arrays (CRITICAL - Gemini requires this)
        - enum for enums
        - description (helps LLM)
        - Nested objects and arrays
        - Optional fields (anyOf with null)

        Strips:
        - Pydantic constraints (minItems, maxLength, etc.)
        - Complex unions (not involving null)
        """
        properties = full_schema.get("properties", {})
        required = full_schema.get("required", [])

        gemini_properties = {}

        for prop_name, prop_info in properties.items():
            # Handle anyOf (common for Optional types like Optional[List[str]])
            if "anyOf" in prop_info:
                # Extract the non-null type from anyOf
                any_of_types = prop_info["anyOf"]
                non_null_type = next((t for t in any_of_types if t.get("type") != "null"), None)

                if non_null_type:
                    # Use the non-null type as base, make it nullable
                    prop_info = non_null_type.copy()
                    prop_info["nullable"] = True
                    # Preserve description from original
                    if "description" in prop_info:
                        pass
                    elif "description" in full_schema.get("properties", {}).get(prop_name, {}):
                        prop_info["description"] = full_schema["properties"][prop_name]["description"]

            prop_type = prop_info.get("type", "string")
            prop_def = {"type": prop_type}

            # Mark as nullable if it was from anyOf with null
            if prop_info.get("nullable"):
                prop_def["nullable"] = True

            # Preserve description (helps LLM understand intent)
            if "description" in prop_info:
                prop_def["description"] = prop_info["description"]

            # CRITICAL: Preserve items for arrays
            if prop_type == "array" and "items" in prop_info:
                items_info = prop_info["items"]

                # Nested object arrays
                if items_info.get("type") == "object":
                    # Recursive conversion for nested objects
                    prop_def["items"] = self._convert_to_gemini_schema(items_info)

                # Simple type arrays
                else:
                    item_type = items_info.get("type", "string")
                    prop_def["items"] = {"type": item_type}

                    # Preserve enum in array items
                    if "enum" in items_info:
                        prop_def["items"]["enum"] = items_info["enum"]

            # Preserve enums
            if "enum" in prop_info:
                prop_def["enum"] = prop_info["enum"]

            # Handle nested objects - recursive conversion
            if prop_type == "object" and "properties" in prop_info:
                nested_schema = self._convert_to_gemini_schema(prop_info)
                prop_def["properties"] = nested_schema["properties"]

                if "required" in prop_info:
                    prop_def["required"] = prop_info["required"]

            gemini_properties[prop_name] = prop_def

        return {
            "type": "object",
            "properties": gemini_properties,
            "required": required
        }
    
    def openai_function_calling(self,
                              messages: list,
                              tools: list = None,
                              model: str = DEFAULT_OAI_MODEL,
                              temperature: float = 0.7,
                              max_tokens: int = 1000,
                              tool_choice: str = "auto"):
        """
        Function calling support for Azure ChatCompletionsClient.
        Returns the full response object to handle tool calls.
        """
        if model == "gpt-4.1":
            secret = get_secret('vibeset/azure_ai_foundry')
            api_key = secret.get('api_key_o3')
            endpoint = (
                "https://kevin-m86fxp36-eastus2.cognitiveservices.azure.com/"
                "openai/deployments/gpt-4.1"
            )
            
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential

            client = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
                api_version="2025-02-01-preview"
            )
            
            # For function calling, use tools parameter if supported
            kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0,
                "model": model
            }
            
            # Add tools if provided and supported
            if tools:
                try:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice
                except:
                    # If tools not supported, fall back to regular completion
                    pass
            
            completion = client.complete(**kwargs)
            return completion
        else:
            # Fallback for other models - create a mock response structure
            if messages:
                last_message = messages[-1]["content"] if messages else ""
                content = self.openai_text(prompt=last_message, model=model, temperature=temperature, max_tokens=max_tokens)
                
                # Create a mock response object that mimics the structure
                class MockChoice:
                    def __init__(self, content):
                        self.message = MockMessage(content)
                
                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                        self.tool_calls = None
                
                class MockResponse:
                    def __init__(self, content):
                        self.choices = [MockChoice(content)]
                
                return MockResponse(content)
    
    @staticmethod
    def get_embedding(
        text: Union[str, List[str]],
        model: str = "text-embedding-3-large",
        dimensions: int = 1024
    ) -> np.ndarray:
        """
        Generate embeddings for a single string or a list of strings using
        Azure's text-embedding-3-large with configurable dimensionality.
        
        Args:
            text: A string or list of strings to embed.
            model: The Azure deployment/model name (default "text-embedding-3-large").
            dimensions: Target embedding size (default 1024).
        
        Returns:
            • np.ndarray shape (dim,)   if input was str
            • np.ndarray shape (N, dim) if input was list[str]
        """
        # 1. Normalize input
        is_single = isinstance(text, str)
        inputs = [text] if is_single else text

        # 2. Fetch API key
        secret = get_secret("vibeset/azure_ai_foundry")
        key = secret["api_key_o3"]

        # 3. Initialize Azure Embeddings client
        endpoint = "https://kevin-m86fxp36-eastus2.cognitiveservices.azure.com/openai/deployments/" + model
        client = EmbeddingsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

        # 4. Call the embed API with dimensions override
        resp = client.embed(
            input=inputs,
            model=model,
            dimensions=dimensions
        )
        embeddings = [item.embedding for item in resp.data]

        # 5. Return as numpy array (flatten if single)
        arr = np.asarray(embeddings, dtype=np.float32)
        return arr[0] if is_single else arr
    
    def openai_image_analysis(self, image_url: str) -> dict:
        """
        Analyze an image for tracklist extraction using OpenAI's vision model.
        
        Args:
            image_url: Base64-encoded image URL or file path
            
        Returns:
            Dict containing tracklist and metadata
        """
        try:
            # Use OpenAI for image analysis
            secret = get_secret("vibeset/azure_ai_foundry")
            key = secret["api_key_o3"]
            
            client = OpenAI(
                api_key=key,
                base_url="https://vibesetbackend9912493372.cognitiveservices.azure.com/openai",
                default_headers={"api-version": "2025-02-01-preview"}
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing images to extract music tracklists. Extract track names and artists from any tracklist, setlist, or music-related image. Return JSON format: {'tracklist': [{'track': 'song name', 'artist': 'artist name'}], 'message': 'description'}. If no tracks found, return empty tracklist with appropriate message."
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this image and extract any tracklist or setlist information. Look for song titles and artist names."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Vision-capable model
                messages=messages,
                max_tokens=1500,
                temperature=0.1
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "tracklist": [],
                    "message": "Could not parse tracklist from image",
                    "raw_response": response_text
                }
                
        except Exception as e:
            return {
                "error": f"Image analysis failed: {str(e)}",
                "tracklist": []
            }
    
    # =============================================================================
    # UNIFIED PROVIDER INTERFACE WITH EVALUATION PLATFORM
    # =============================================================================
    
    def generate_structured(self,
                          prompt: str,
                          schema_name: str,
                          provider: str = "azure",
                          model: str = None,
                          collect_tokens: bool = False,
                          **kwargs) -> str:
        """
        Unified structured generation with equal provider treatment and evaluation metrics.
        
        This is the main method for V5 algorithm evaluation platform.
        
        Args:
            prompt: The input prompt
            prompt: The input prompt
            schema_name: Name of schema OR dict/schema object directly
            provider: "azure" or "gemini"
            model: Specific model or None for provider default
            collect_tokens: Whether to collect latency/performance metrics
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            JSON string with structured output
            
        Example:
            # Azure with specific model
            result = gai.generate_structured(
                prompt="Create a house setlist",
                schema_name="ChunkedSetlistSchema",
                provider="azure",
                model="gpt-4.1"
            )
            
            # Gemini with thinking
            result = gai.generate_structured(
                prompt="Create a house setlist", 
                schema_name="ChunkedSetlistSchema",
                provider="gemini",
                model="gemini-2.5-flash",
                thinking_level="medium"
            )
        """

        import time
        start_time = time.time() if collect_tokens else None

        # Get provider-optimized schema
        if isinstance(schema_name, (dict, type)) or hasattr(schema_name, 'model_json_schema'):
             schema = schema_name
        else:
             schema = self._get_optimized_schema(schema_name, provider)
        
        # Auto-select model if not specified
        if not model:
            model = self._get_default_model(provider)
        
        # Provider-specific generation leveraging each provider's strengths
        try:
            if provider == "azure":
                result = self._azure_structured_generate(prompt, schema, model, collect_tokens, **kwargs)
            elif provider == "gemini":
                result = self._gemini_structured_generate(prompt, schema, model, collect_tokens, **kwargs)
            elif provider == "perplexity":
                result = self._perplexity_structured_generate(prompt, schema, model, collect_tokens, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Handle different return types (string or dict with token data)
            if isinstance(result, dict) and 'response' in result:
                # Token collection enabled - result is a dict
                response_text = result['response']

                # Enhanced metrics collection with token/cost data
                if collect_tokens and start_time:
                    self._log_evaluation_metrics(
                        provider=provider,
                        model=model,
                        schema_name=schema_name,
                        latency=time.time() - start_time,
                        prompt_length=len(prompt),
                        response_length=len(response_text) if response_text else 0,
                        token_data=result  # Pass full token/cost data
                    )
                
                return response_text
            else:
                # Regular string result
                if collect_tokens and start_time:
                    self._log_evaluation_metrics(
                        provider=provider,
                        model=model,
                        schema_name=schema_name,
                        latency=time.time() - start_time,
                        prompt_length=len(prompt),
                        response_length=len(result) if result else 0
                    )
                
                return result
            
        except Exception as e:
            logging.error(f"Structured generation failed - provider: {provider}, model: {model}, schema: {schema_name}, error: {e}")
            raise RuntimeError(f"{provider} structured generation error: {str(e)}")
    
    def _get_optimized_schema(self, schema_name: str, provider: str):
        """Get schema optimized for each provider's strengths"""
        try:
            from schemas.registry import SchemaRegistry
            print(f"🔍 Debug: Using schema registry for {schema_name}")
            return SchemaRegistry.get_schema(schema_name, provider)
        except ImportError:
            # Fallback for environments without schema registry
            print(f"🔍 Debug: Schema registry not available, using fallback for {schema_name}")
            return self._fallback_get_schema(schema_name, provider)
        except Exception as e:
            print(f"🔍 Debug: Schema registry failed for {schema_name}: {e}")
            return self._fallback_get_schema(schema_name, provider)
    
    def _fallback_get_schema(self, schema_name: str, provider: str):
        """Fallback schema retrieval from prompts_params.py"""
        try:
            import prompts_params as pp
            if not hasattr(pp, schema_name):
                available_schemas = [attr for attr in dir(pp) if 'Schema' in attr]
                raise ValueError(f"Schema {schema_name} not found. Available: {available_schemas}")
                
            base_schema = getattr(pp, schema_name)
            
            if provider == "azure":
                from azure.ai.inference.models._models import JsonSchemaFormat
                return JsonSchemaFormat(schema=base_schema.model_json_schema(), name=schema_name)
            else:
                return base_schema
        except Exception as e:
            raise ValueError(f"Could not retrieve schema {schema_name}: {e}")
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for each provider"""
        defaults = {
            "azure": DEFAULT_OAI_MODEL,        # "gpt-4.1"
            "gemini": DEFAULT_GEMINI_MODEL,    # "gemini-2.5-flash"
            "perplexity": DEFAULT_PERPLEXITY_MODEL  # "sonar"
        }
        return defaults.get(provider, DEFAULT_OAI_MODEL)
    
    def _azure_structured_generate(self, prompt: str, schema, model: str, collect_tokens: bool = False, **kwargs) -> Union[str, dict]:
        """Azure-optimized structured generation"""
        # Extract Azure-specific parameters
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 4095)
        sysmsg = kwargs.get('sysmsg', None)
        
        return self.openai_text(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            sysmsg=sysmsg,
            to_json=True,
            json_schema=schema,  # Already JsonSchemaFormat from registry
            collect_tokens=collect_tokens  # Enable token collection if metrics requested
        )
    
    def _gemini_structured_generate(self, prompt: str, schema, model: str, collect_tokens: bool = False, **kwargs) -> Union[str, dict]:
        """Gemini-optimized structured generation with thinking"""
        # Extract Gemini-specific parameters  
        temperature = kwargs.get('temperature', 0.3)
        max_tokens = kwargs.get('max_tokens', 4095)
        sysmsg = kwargs.get('sysmsg', None)
        thinking_level = kwargs.get('thinking_level', 'simple')
        
        return self.gemini_text(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            sysmsg=sysmsg,
            to_json=True,
            json_schema=schema,  # Direct Pydantic model from registry
            thinking_level=thinking_level,
            collect_tokens=collect_tokens  # Enable token collection if metrics requested
        )

    def _perplexity_structured_generate(self, prompt: str, schema, model: str, collect_tokens: bool = False, **kwargs) -> Union[str, dict]:
        """Perplexity-optimized structured generation with search capabilities"""
        # Extract Perplexity-specific parameters
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 4095)
        sysmsg = kwargs.get('sysmsg', None)
        search_domain_filter = kwargs.get('search_domain_filter', None)
        search_recency_filter = kwargs.get('search_recency_filter', None)
        return_images = kwargs.get('return_images', False)
        return_related_questions = kwargs.get('return_related_questions', False)
        search_mode = kwargs.get('search_mode', 'web')

        # Convert schema to dict format for Perplexity (similar to OpenAI)
        json_schema_dict = None
        if hasattr(schema, 'model_json_schema'):
            # Pydantic model - convert to JSON schema
            json_schema_dict = schema.model_json_schema()
        elif hasattr(schema, 'schema'):
            # Azure JsonSchemaFormat - extract schema
            json_schema_dict = schema.schema
        elif isinstance(schema, dict):
            # Already a dict
            json_schema_dict = schema

        return self.perplexity_text(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            sysmsg=sysmsg,
            to_json=True,
            json_schema=json_schema_dict,
            collect_tokens=collect_tokens,
            search_domain_filter=search_domain_filter,
            search_recency_filter=search_recency_filter,
            return_images=return_images,
            return_related_questions=return_related_questions,
            search_mode=search_mode
        )

    async def _async_perplexity_structured_generate(self, prompt: str, schema, model: str, collect_tokens: bool = False, **kwargs) -> Union[str, dict]:
        """Async Perplexity-optimized structured generation with search capabilities"""
        # Extract Perplexity-specific parameters
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 4095)
        sysmsg = kwargs.get('sysmsg', None)
        search_domain_filter = kwargs.get('search_domain_filter', None)
        search_recency_filter = kwargs.get('search_recency_filter', None)
        return_images = kwargs.get('return_images', False)
        return_related_questions = kwargs.get('return_related_questions', False)
        search_mode = kwargs.get('search_mode', 'web')

        # Convert schema to dict format for Perplexity (similar to OpenAI)
        json_schema_dict = None
        if hasattr(schema, 'model_json_schema'):
            # Pydantic model - convert to JSON schema
            json_schema_dict = schema.model_json_schema()
        elif hasattr(schema, 'schema'):
            # Azure JsonSchemaFormat - extract schema
            json_schema_dict = schema.schema
        elif isinstance(schema, dict):
            # Already a dict
            json_schema_dict = schema

        return await self.async_perplexity_text(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            sysmsg=sysmsg,
            to_json=True,
            json_schema=json_schema_dict,
            collect_tokens=collect_tokens,
            search_domain_filter=search_domain_filter,
            search_recency_filter=search_recency_filter,
            return_images=return_images,
            return_related_questions=return_related_questions,
            search_mode=search_mode
        )

    async def async_generate_structured(self,
                                       prompt: str,
                                       schema_name: str,
                                       provider: str = "azure",
                                       model: str = None,
                                       collect_tokens: bool = False,
                                       **kwargs) -> str:
        """
        Async version of generate_structured for true parallel execution.

        Enables concurrent LLM calls in A/B testing and parallel algorithm components.
        Currently supports Azure only; Gemini falls back to sync execution.

        Args:
            prompt: The input prompt
            prompt: The input prompt
            schema_name: Name of schema OR dict/schema object directly
            provider: "azure" or "gemini"
            model: Specific model or None for provider default
            collect_tokens: Whether to collect latency/performance metrics
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            JSON string with structured output

        Example:
            # Run 3 Azure calls in parallel
            results = await asyncio.gather(
                gai.async_generate_structured(prompt1, "Schema1", provider="azure"),
                gai.async_generate_structured(prompt2, "Schema2", provider="azure"),
                gai.async_generate_structured(prompt3, "Schema3", provider="azure")
            )
        """
        import time
        start_time = time.time() if collect_tokens else None

        # Get provider-optimized schema
        if isinstance(schema_name, (dict, type)) or hasattr(schema_name, 'model_json_schema'):
             schema = schema_name
        else:
             schema = self._get_optimized_schema(schema_name, provider)

        # Auto-select model if not specified
        if not model:
            model = self._get_default_model(provider)

        # Provider-specific async generation
        try:
            if provider == "azure":
                result = await self._async_azure_structured_generate(prompt, schema, model, collect_tokens, **kwargs)
            elif provider == "gemini":
                # Gemini async not yet implemented - fall back to sync
                logging.warning("Gemini async not implemented, falling back to sync execution")
                result = self._gemini_structured_generate(prompt, schema, model, collect_tokens, **kwargs)
            elif provider == "perplexity":
                result = await self._async_perplexity_structured_generate(prompt, schema, model, collect_tokens, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Handle different return types (string or dict with token data)
            if isinstance(result, dict) and 'response' in result:
                # Token collection enabled - result is a dict
                response_text = result['response']

                # Enhanced metrics collection with token/cost data
                if collect_tokens and start_time:
                    self._log_evaluation_metrics(
                        provider=provider,
                        model=model,
                        schema_name=schema_name,
                        latency=time.time() - start_time,
                        prompt_length=len(prompt),
                        response_length=len(response_text) if response_text else 0,
                        token_data=result  # Pass full token/cost data
                    )

                return response_text
            else:
                # Regular string result
                if collect_tokens and start_time:
                    self._log_evaluation_metrics(
                        provider=provider,
                        model=model,
                        schema_name=schema_name,
                        latency=time.time() - start_time,
                        prompt_length=len(prompt),
                        response_length=len(result) if result else 0
                    )

                return result

        except Exception as e:
            logging.error(f"Async structured generation failed - provider: {provider}, model: {model}, schema: {schema_name}, error: {e}")
            raise RuntimeError(f"{provider} async structured generation error: {str(e)}")

    async def _async_azure_structured_generate(self, prompt: str, schema, model: str, collect_tokens: bool = False, **kwargs) -> Union[str, dict]:
        """Azure-optimized async structured generation"""
        # Extract Azure-specific parameters
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 4095)
        sysmsg = kwargs.get('sysmsg', None)
        strict = kwargs.get('strict', False)

        return await self.async_openai_text(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            sysmsg=sysmsg,
            to_json=True,
            json_schema=schema,  # Already JsonSchemaFormat from registry or dict
            collect_tokens=collect_tokens,  # Enable token collection if metrics requested
            strict=strict
        )

    def _log_evaluation_metrics(self, provider: str, model: str, schema_name: str, 
                              latency: float, prompt_length: int, response_length: int, token_data: dict = None):
        """Log evaluation metrics for A/B testing analysis"""
        import time
        metrics = {
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "schema_name": schema_name,
            "latency_seconds": latency,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "tokens_per_second": response_length / latency if latency > 0 else 0
        }
        
        # Add token/cost data if available
        if token_data:
            usage = token_data.get('usage', {})
            metrics.update({
                "input_tokens": usage.get('input_tokens', 0),
                "output_tokens": usage.get('output_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0),
                "cost_usd": token_data.get('cost', 0.0),
                "cost_per_token": token_data.get('cost', 0.0) / max(usage.get('total_tokens', 1), 1)
            })
        
        # Log to console for now - can be enhanced with proper metrics collection
        logging.info(f"LLM_METRICS: {metrics}")
        
        # Future: Send to monitoring system (CloudWatch, DataDog, etc.)
        # self._send_to_monitoring_system(metrics)

# =============================================================================
# MODEL PRICING AND COST TRACKING
# =============================================================================

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ModelPricing:
    """Pricing information for a model"""
    model_name: str
    provider: str
    input_price_per_1k_tokens: float  # USD per 1K input tokens
    output_price_per_1k_tokens: float  # USD per 1K output tokens
    notes: str = ""

class PricingRegistry:
    """Registry of current model pricing information (August 2025)"""
    
    # Current pricing based on official API documentation
    PRICING_DATA = {
        # Azure OpenAI pricing (current 2025 rates)
        "azure_gpt-4.1": ModelPricing(
            model_name="gpt-4.1",
            provider="azure",
            input_price_per_1k_tokens=0.002,   # Current Azure GPT-4.1 rate
            output_price_per_1k_tokens=0.008,  # Current Azure GPT-4.1 rate
            notes="Azure OpenAI GPT-4.1 latest model"
        ),
        "azure_gpt-4": ModelPricing(
            model_name="gpt-4",
            provider="azure", 
            input_price_per_1k_tokens=0.010,   # Standard GPT-4 Turbo rate
            output_price_per_1k_tokens=0.030,  # Standard GPT-4 Turbo rate
            notes="Azure OpenAI GPT-4 Turbo"
        ),
        "azure_gpt-3.5-turbo": ModelPricing(
            model_name="gpt-3.5-turbo",
            provider="azure",
            input_price_per_1k_tokens=0.0005,  # GPT-3.5 Turbo rate
            output_price_per_1k_tokens=0.0015, # GPT-3.5 Turbo rate
            notes="Azure OpenAI GPT-3.5 Turbo"
        ),
        "azure_gpt-4o": ModelPricing(
            model_name="gpt-4o",
            provider="azure",
            input_price_per_1k_tokens=0.005,   # $5.00 per 1M tokens
            output_price_per_1k_tokens=0.015,  # $15.00 per 1M tokens
            notes="Azure OpenAI GPT-4o"
        ),
        "azure_gpt-4o-mini": ModelPricing(
            model_name="gpt-4o-mini",
            provider="azure",
            input_price_per_1k_tokens=0.00015, # $0.15 per 1M tokens
            output_price_per_1k_tokens=0.0006, # $0.60 per 1M tokens
            notes="Azure OpenAI GPT-4o mini"
        ),
        "azure_gpt-4.1-mini": ModelPricing(
            model_name="gpt-4.1-mini",
            provider="azure",
            input_price_per_1k_tokens=0.0015,  # Cost-optimized mini pricing
            output_price_per_1k_tokens=0.006,  # Cost-optimized mini pricing
            notes="Azure OpenAI GPT-4.1 mini"
        ),
        "azure_gpt-4.1-nano": ModelPricing(
            model_name="gpt-4.1-nano",
            provider="azure",
            input_price_per_1k_tokens=0.0005, # Ultra-low cost tier
            output_price_per_1k_tokens=0.002, # Ultra-low cost tier
            notes="Azure OpenAI GPT-4.1 nano - ultra cost-optimized"
        ),
        "azure_o4-mini": ModelPricing(
            model_name="o4-mini",
            provider="azure",
            input_price_per_1k_tokens=0.0005,  # Estimated competitive pricing
            output_price_per_1k_tokens=0.002,  # Estimated competitive pricing
            notes="Azure OpenAI o4-mini preview"
        ),
        "azure_gpt-5-mini": ModelPricing(
            model_name="gpt-5-mini",
            provider="azure",
            input_price_per_1k_tokens=0.0003,  # Estimated competitive pricing
            output_price_per_1k_tokens=0.0012,  # Estimated competitive pricing
            notes="Azure OpenAI GPT-5 mini - cost-optimized reasoning"
        ),
        "azure_gpt-5-nano": ModelPricing(
            model_name="gpt-5-nano",
            provider="azure",
            input_price_per_1k_tokens=0.0001,  # Ultra-low cost
            output_price_per_1k_tokens=0.0004,  # Ultra-low cost
            notes="Azure OpenAI GPT-5 nano - ultra cost-optimized"
        ),
        "azure_mistral-small-2503": ModelPricing(
            model_name="mistral-small-2503",
            provider="azure",
            input_price_per_1k_tokens=0.0002,  # Est. ~$0.20 per 1M (Mistral Small is cheap)
            output_price_per_1k_tokens=0.0006, # Est. ~$0.60 per 1M
            notes="Mistral Small 2503 (Azure MaaS)"
        ),

        # Google Gemini pricing (official API rates as of August 2025)
        "gemini_gemini-2.5-pro": ModelPricing(
            model_name="gemini-2.5-pro",
            provider="gemini",
            input_price_per_1k_tokens=0.00125,  # $1.25 per 1M tokens (≤200k context)
            output_price_per_1k_tokens=0.010,   # $10.00 per 1M tokens (≤200k context)
            notes="Google Gemini 2.5 Pro - premium model with thinking"
        ),
        "gemini_gemini-2.5-flash": ModelPricing(
            model_name="gemini-2.5-flash",
            provider="gemini",
            input_price_per_1k_tokens=0.0003,   # $0.30 per 1M tokens
            output_price_per_1k_tokens=0.0025,  # $2.50 per 1M tokens
            notes="Google Gemini 2.5 Flash - balanced performance with thinking"
        ),
        "gemini_gemini-1.5-pro": ModelPricing(
            model_name="gemini-1.5-pro",
            provider="gemini",
            input_price_per_1k_tokens=0.00125,  # $1.25 per 1M tokens (≤128k context)
            output_price_per_1k_tokens=0.005,   # $5.00 per 1M tokens (≤128k context)
            notes="Google Gemini 1.5 Pro - previous generation"
        ),
        "gemini_gemini-2.5-flash-lite": ModelPricing(
            model_name="gemini-2.5-flash-lite",
            provider="gemini",
            input_price_per_1k_tokens=0.0001,   # Estimated ultra-low cost
            output_price_per_1k_tokens=0.0005,  # Estimated ultra-low cost
            notes="Google Gemini 2.5 Flash Lite - ultra cost-optimized"
        ),

        # Perplexity Sonar API pricing (hybrid token + request model)
        "perplexity_sonar": ModelPricing(
            model_name="sonar",
            provider="perplexity",
            input_price_per_1k_tokens=0.001,    # $1.00 per 1M tokens
            output_price_per_1k_tokens=0.001,   # $1.00 per 1M tokens
            notes="Perplexity Sonar - lightweight search + $5/1K requests"
        ),
        "perplexity_sonar-pro": ModelPricing(
            model_name="sonar-pro",
            provider="perplexity",
            input_price_per_1k_tokens=0.001,    # $1.00 per 1M tokens
            output_price_per_1k_tokens=0.001,   # $1.00 per 1M tokens
            notes="Perplexity Sonar Pro - advanced search + $8/1K requests"
        ),
        "perplexity_sonar-reasoning": ModelPricing(
            model_name="sonar-reasoning",
            provider="perplexity",
            input_price_per_1k_tokens=0.001,    # $1.00 per 1M tokens
            output_price_per_1k_tokens=0.001,   # $1.00 per 1M tokens
            notes="Perplexity Sonar Reasoning - fast reasoning + $12/1K requests"
        ),
        "perplexity_sonar-reasoning-pro": ModelPricing(
            model_name="sonar-reasoning-pro",
            provider="perplexity",
            input_price_per_1k_tokens=0.001,    # $1.00 per 1M tokens (estimated)
            output_price_per_1k_tokens=0.001,   # $1.00 per 1M tokens (estimated)
            notes="Perplexity Sonar Reasoning Pro - precise CoT reasoning"
        ),
        "perplexity_sonar-deep-research": ModelPricing(
            model_name="sonar-deep-research",
            provider="perplexity",
            input_price_per_1k_tokens=0.002,    # Estimated premium pricing
            output_price_per_1k_tokens=0.002,   # Estimated premium pricing
            notes="Perplexity Sonar Deep Research - expert-level analysis"
        )
    }
    
    @classmethod
    def get_pricing(cls, provider: str, model: str) -> Optional[ModelPricing]:
        """Get pricing for a specific model"""
        key = f"{provider}_{model}"
        return cls.PRICING_DATA.get(key)
    
    @classmethod
    def calculate_cost(cls, provider: str, model: str, input_tokens: int, output_tokens: int, request_count: int = 1) -> float:
        """Calculate cost for token usage and requests"""
        pricing = cls.get_pricing(provider, model)
        if not pricing:
            return 0.0

        # Base token cost
        input_cost = (input_tokens / 1000) * pricing.input_price_per_1k_tokens
        output_cost = (output_tokens / 1000) * pricing.output_price_per_1k_tokens
        token_cost = input_cost + output_cost

        # Handle Perplexity's hybrid pricing (tokens + request fees)
        if provider == "perplexity":
            request_cost_per_1k = {
                "sonar": 5.0,                    # $5 per 1K requests
                "sonar-pro": 8.0,               # $8 per 1K requests
                "sonar-reasoning": 12.0,        # $12 per 1K requests
                "sonar-reasoning-pro": 15.0,    # Estimated $15 per 1K requests
                "sonar-deep-research": 20.0     # Estimated $20 per 1K requests
            }
            request_fee = (request_count / 1000) * request_cost_per_1k.get(model, 5.0)
            return token_cost + request_fee

        return token_cost
    
    @classmethod
    def get_all_pricing(cls) -> Dict[str, ModelPricing]:
        """Get all pricing information"""
        return cls.PRICING_DATA.copy()
    
    @classmethod
    def print_pricing_table(cls):
        """Print formatted pricing table"""
        
        print("💰 MODEL PRICING TABLE (USD)")
        print("=" * 80)
        print(f"{'Provider':<10} {'Model':<20} {'Input ($/1K)':<12} {'Output ($/1K)':<13} {'Notes'}")
        print("-" * 80)
        
        # Group by provider
        azure_models = []
        gemini_models = []
        perplexity_models = []

        for pricing in cls.PRICING_DATA.values():
            if pricing.provider == "azure":
                azure_models.append(pricing)
            elif pricing.provider == "gemini":
                gemini_models.append(pricing)
            elif pricing.provider == "perplexity":
                perplexity_models.append(pricing)
        
        # Print Azure models
        for i, pricing in enumerate(azure_models):
            provider_name = "Azure" if i == 0 else ""
            print(f"{provider_name:<10} {pricing.model_name:<20} ${pricing.input_price_per_1k_tokens:<11.6f} ${pricing.output_price_per_1k_tokens:<12.6f} {pricing.notes}")
        
        print()
        
        # Print Gemini models
        for i, pricing in enumerate(gemini_models):
            provider_name = "Gemini" if i == 0 else ""
            print(f"{provider_name:<10} {pricing.model_name:<20} ${pricing.input_price_per_1k_tokens:<11.6f} ${pricing.output_price_per_1k_tokens:<12.6f} {pricing.notes}")

        print()

        # Print Perplexity models
        for i, pricing in enumerate(perplexity_models):
            provider_name = "Perplexity" if i == 0 else ""
            print(f"{provider_name:<10} {pricing.model_name:<20} ${pricing.input_price_per_1k_tokens:<11.6f} ${pricing.output_price_per_1k_tokens:<12.6f} {pricing.notes}")

        print()

    @classmethod
    def detect_provider_for_model(cls, model: str) -> str:
        """
        Backwards-compatible provider detection for model names.

        This ensures Azure models always use Azure pricing, regardless of
        how they're called (sync vs async).

        Args:
            model: Model name (e.g., 'gpt-4.1', 'gpt-4.1-mini', 'sonar')

        Returns:
            Provider name ('azure', 'gemini', 'perplexity')
        """

        # Perplexity models
        perplexity_models = [
            'sonar', 'sonar-pro', 'sonar-reasoning', 'sonar-deep-research'
        ]

        # Check Azure models first (most critical)
        if model in azure_models:
            return 'azure'

        # Check Gemini models
        if model in gemini_models:
            return 'gemini'

        # Check Perplexity models
        if model in perplexity_models:
            return 'perplexity'

        # Default to Azure for unknown models (conservative approach)
        # This prevents accidentally using wrong pricing
        return 'azure'

class CostTracker:
    """Track costs during A/B testing"""
    
    def __init__(self):
        self.test_costs = []
    
    def add_test_cost(self, test_id: str, provider: str, model: str, 
                     input_tokens: int, output_tokens: int, execution_time: float):
        """Add cost data for a test"""
        
        pricing = PricingRegistry.get_pricing(provider, model)
        if pricing:
            cost = PricingRegistry.calculate_cost(provider, model, input_tokens, output_tokens)
            cost_per_second = cost / execution_time if execution_time > 0 else 0
            
            self.test_costs.append({
                "test_id": test_id,
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "execution_time": execution_time,
                "total_cost": cost,
                "cost_per_second": cost_per_second,
                "cost_per_1k_tokens": cost / ((input_tokens + output_tokens) / 1000) if (input_tokens + output_tokens) > 0 else 0
            })
    
    def get_cost_summary(self) -> Dict[str, float]:
        """Get cost summary by provider/model"""
        
        summary = {}
        
        for cost_data in self.test_costs:
            key = f"{cost_data['provider']}_{cost_data['model']}"
            if key not in summary:
                summary[key] = {
                    "total_cost": 0,
                    "total_tokens": 0,
                    "test_count": 0,
                    "total_time": 0
                }
            
            summary[key]["total_cost"] += cost_data["total_cost"]
            summary[key]["total_tokens"] += cost_data["total_tokens"]
            summary[key]["test_count"] += 1
            summary[key]["total_time"] += cost_data["execution_time"]
        
        return summary
    
    def print_cost_analysis(self):
        """Print detailed cost analysis"""
        
        if not self.test_costs:
            print("No cost data available")
            return
        
        summary = self.get_cost_summary()
        
        print("\n💰 COST ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"{'Model':<25} {'Tests':<6} {'Total $':<10} {'Avg $/test':<12} {'$/1K tokens'}")
        print("-" * 60)
        
        total_cost = 0
        for model_key, data in summary.items():
            provider, model = model_key.split("_", 1)
            avg_cost_per_test = data["total_cost"] / data["test_count"]
            avg_cost_per_1k_tokens = data["total_cost"] / (data["total_tokens"] / 1000) if data["total_tokens"] > 0 else 0
            
            print(f"{model:<25} {data['test_count']:<6} ${data['total_cost']:<9.6f} ${avg_cost_per_test:<11.6f} ${avg_cost_per_1k_tokens:.6f}")
            total_cost += data["total_cost"]
        
        print("-" * 60)
        print(f"{'TOTAL':<25} {'':<6} ${total_cost:<9.6f}")
        print()

def estimate_setlist_tokens(setlist_length_minutes: int, complexity: str = "medium") -> Dict[str, int]:
    """Estimate token usage for setlist generation"""
    
    # Base estimates for different complexities
    complexity_multipliers = {
        "simple": 0.7,
        "medium": 1.0,
        "complex": 1.5,
        "extreme": 2.0
    }
    
    multiplier = complexity_multipliers.get(complexity, 1.0)
    
    # Base token estimates (tuned from actual usage)
    base_input = 2000 + (setlist_length_minutes * 50)  # Prompt + context
    base_output = 5000 + (setlist_length_minutes * 200)  # Generated setlist
    
    return {
        "input": int(base_input * multiplier),
        "output": int(base_output * multiplier)
    }