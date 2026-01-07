
import os
import json
import google.generativeai as genai
from pathlib import Path
import time

# Try simple manual load first
env_path = Path(__file__).resolve().parents[2] / '.env'
api_key = None
try:
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
except Exception as e:
    print(f"[WARN] Manual .env read failed: {e}")

# Fallback to os.getenv if manual failed (or if key is in system env)
if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

class LLMParser:
    def __init__(self):
        self.api_key = api_key
        
        if not self.api_key:
            print("[WARN] WARNING: GEMINI_API_KEY not found in .env")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('models/gemini-2.5-flash')
                print("Gemini 2.5 Flash Model Loaded.")
            except Exception as e:
                print(f"Gemini Setup Error: {e}")
                self.model = None

    def parse_query(self, user_query):
        """
        Input: "Find me a video platform using Python and Go"
        Output: {
            "query": "video platform",
            "tech_stack": ["Python", "Go"],
            "tags": []
        }
        """
        if not self.model:
            return {"query": user_input, "tech_stack": [], "tags": []}

        prompt = f"""
        You are an AI Search Assistant. User will send a request to find a software project.
        Your task is to extract:
        Analyze this search query for a software project database.
        Query: "{user_query}"
        
        Extract:
        1. "query": The core semantic meaning (translate to English if necessary for better search).
        2. "tech_stack": List of specific technologies mentioned (e.g., React, Python).
        3. "tags": List of implied categories (e.g., "social", "e-commerce", "mobile", "web", "ai").
        
        Return JSON ONLY. Example:
        {{"query": "social network project", "tech_stack": ["React"], "tags": ["social", "web"]}}
        """
        
        retries = 3
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                
                # Clean markdown code blocks
                clean_text = response.text.strip()
                if clean_text.startswith("```"):
                    clean_text = clean_text.replace("```json", "").replace("```", "")
                    
                parsed = json.loads(clean_text)
                
                # Normalize
                if 'tags' in parsed:
                    parsed['tags'] = [t.lower() for t in parsed['tags']]
                    
                return parsed
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (2 ** attempt) * 10  # Increased: 10s, 20s, 40s
                    print(f"LLM Rate Limit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"LLM Error: {e}")
                    break
                    
        # Fallback if AI fails
        return {
            "query": user_query,
            "tech_stack": [],
            "tags": []
        }

    def parse_query_v2(self, user_query):
        """
        Enhanced parser for Schema V2 - Extract structured filters
        
        Input: "Find React ecommerce site with medium complexity"
        Output: {
            "intent": "find_project",
            "filters": {
                "domain": "ecommerce",
                "frontend": ["React"],
                "backend": [],
                "platform": [],
                "complexity": "medium",
                "tags": []
            },
            "semantic_query": "ecommerce platform"
        }
        """
        if not self.model:
            return {
                "intent": "find_project",
                "filters": {},
                "semantic_query": user_query
            }

        prompt = f"""
You are a Search Query Analyzer. Extract structured filters from user queries.

Query: "{user_query}"

Return STRICT JSON with this schema:
{{
  "intent": "find_project",
  "filters": {{
    "domain": "ecommerce" | "social" | "fintech" | "education" | "media" | "communication" | "productivity" | "cloud" | null,
    "frontend": ["React", "Vue.js"] | [],
    "backend": ["Node.js", "Python"] | [],
    "platform": ["web", "mobile", "desktop"] | [],
    "complexity": "low" | "medium" | "high" | null,
    "tags": ["b2c", "real-time"] | []
  }},
  "semantic_query": "simplified query for vector search"
}}

Extraction Rules:
1. domain: Map keywords to domains
   - "ecommerce", "shop", "marketplace" → "ecommerce"
   - "social", "network", "chat" → "social"
   - "payment", "bank" → "fintech"
   - "video", "music", "streaming" → "media"
   
2. frontend/backend: Extract specific tech names
   - Frontend: React, Vue.js, Angular, Flutter, React Native, Svelte
   - Backend: Node.js, Python, Java, Go, PHP, Ruby, C#, Erlang
   
3. platform: Infer from context
   - "mobile app" → ["mobile"]
   - "website" → ["web"]
   - "desktop" → ["desktop"]
   
4. complexity: Detect keywords
   - "simple", "basic" → "low"
   - "medium", "moderate" → "medium"
   - "complex", "advanced", "enterprise" → "high"
   
6. semantic_query: Write a detailed, descriptive query for vector search.
   - Do NOT just remove keywords.
   - EXPAND the query with relevant context and synonyms.
   - Describe the functionality, purpose, or nature of the project.

Examples:
- "React ecommerce site" 
  → domain:"ecommerce", frontend:["React"], semantic_query:"online shopping marketplace platform with product catalog and checkout"
- "Social network phức tạp" 
  → domain:"social", complexity:"high", semantic_query:"large-scale social networking platform with real-time scaling and high user load"
- "Mobile app dùng Flutter" 
  → platform:["mobile"], frontend:["Flutter"], semantic_query:"cross-platform mobile application development"
- "Video streaming với Python" 
  → domain:"media", backend:["Python"], semantic_query:"video streaming service, content delivery platform, media transcoding"
- "tìm dự án giống amazon"
  → domain:"ecommerce", semantic_query:"massive online retailer platform, global marketplace, logistics and supply chain"

Important: Return ONLY valid JSON, no explanations.
"""

        retries = 3
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                
                # Clean response
                clean_text = response.text.strip()
                if clean_text.startswith("```"):
                    clean_text = clean_text.replace("```json", "").replace("```", "").strip()
                
                parsed = json.loads(clean_text)
                
                # Validate and normalize
                if 'filters' not in parsed:
                    parsed['filters'] = {}
                if 'semantic_query' not in parsed:
                    parsed['semantic_query'] = user_query
                if 'intent' not in parsed:
                    parsed['intent'] = "find_project"
                
                # Normalize complexity
                if parsed['filters'].get('complexity'):
                    parsed['filters']['complexity'] = parsed['filters']['complexity'].lower()
                
                return parsed
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (2 ** attempt) * 10
                    print(f"LLM Rate Limit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"LLM Parse V2 Error: {e}")
                    break
        
        # Fallback
        return {
            "intent": "find_project",
            "filters": {},
            "semantic_query": user_query
        }

if __name__ == "__main__":
    parser = LLMParser()
    res = parser.parse_query("Tìm cho tôi mạng xã hội giống Facebook code bằng React")
    print("Parsed Result:", res)

