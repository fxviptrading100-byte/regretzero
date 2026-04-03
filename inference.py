#!/usr/bin/env python3
"""
RegretZero Inference Server - Hackathon Compliant
Uses OpenAI Client with structured logging and JSON responses
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# OpenAI Client Import
try:
    from openai import OpenAI
except ImportError:
    print("[START] ERROR: OpenAI library not found")
    print("[STEP] Installing openai...")
    os.system("pip install openai")
    from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegretZeroInference:
    """Main inference class for RegretZero decision recommendations."""
    
    def __init__(self):
        """Initialize inference with OpenAI client."""
        self.client = None
        self.setup_openai_client()
        
    def setup_openai_client(self):
        """Setup OpenAI client from environment variables."""
        try:
            api_base = os.getenv('API_BASE_URL')
            model_name = os.getenv('MODEL_NAME')
            hf_token = os.getenv('HF_TOKEN')
            
            if not all([api_base, model_name, hf_token]):
                print("[START] ERROR: Missing required environment variables")
                print("[STEP] Required: API_BASE_URL, MODEL_NAME, HF_TOKEN")
                return False
                
            print(f"[START] Initializing OpenAI client with model: {model_name}")
            self.client = OpenAI(
                api_key=hf_token,
                base_url=api_base
            )
            print(f"[STEP] OpenAI client initialized successfully")
            return True
            
        except Exception as e:
            print(f"[START] ERROR: Failed to initialize OpenAI client: {e}")
            return False
    
    def analyze_decision(self, decision_input: str) -> Dict[str, Any]:
        """Analyze decision and return structured response."""
        print(f"[STEP] Analyzing decision: {decision_input[:50]}...")
        
        try:
            # Create prompt for decision analysis
            prompt = f"""
            As a RegretZero AI advisor, analyze this decision: "{decision_input}"
            
            Provide analysis in JSON format with:
            - suggestion: recommended action (1-8)
            - regret_risk: low/medium/high
            - confidence: 0-100
            - reasoning: brief explanation
            - verdict: final recommendation
            
            Actions:
            1. Suggest specific action
            2. Wait and gather information  
            3. Research thoroughly
            4. Talk to experienced person
            5. Reflect deeply
            6. Act immediately
            7. Delegate or ask help
            8. Bring stakeholders together
            """
            
            response = self.client.chat.completions.create(
                model=os.getenv('MODEL_NAME'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"[STEP] Received analysis from model")
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                print(f"[STEP] Successfully parsed JSON response")
                return result
            except json.JSONDecodeError:
                # Fallback structured response
                print("[STEP] JSON parse failed, using fallback structure")
                return {
                    "suggestion": 1,
                    "regret_risk": "medium",
                    "confidence": 75,
                    "reasoning": "Analysis completed with structured fallback",
                    "verdict": "Consider gathering more information"
                }
                
        except Exception as e:
            print(f"[STEP] ERROR: Analysis failed: {e}")
            return {
                "suggestion": 1,
                "regret_risk": "high",
                "confidence": 50,
                "reasoning": f"Analysis error: {str(e)}",
                "verdict": "System error detected"
            }
    
    def reset(self) -> Dict[str, Any]:
        """Handle reset command and return status."""
        print("[START] Reset command received")
        print("[STEP] Clearing cache and reinitializing...")
        
        # Reinitialize client
        self.setup_openai_client()
        
        status = {
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
            "message": "RegretZero inference system reset successfully"
        }
        
        print(f"[STEP] Reset completed: {status['status']}")
        return status

def main():
    """Main inference server function."""
    print("[START] RegretZero Inference Server Starting...")
    print("[STEP] Initializing components...")
    
    # Initialize inference system
    inference = RegretZeroInference()
    
    if not inference.client:
        print("[END] Failed to initialize - cannot continue")
        sys.exit(1)
    
    print("[STEP] System ready for decision analysis")
    print("[END] RegretZero Inference Server Ready")
    
    # Main interaction loop
    try:
        while True:
            print("\n" + "="*50)
            print("RegretZero Decision Analysis - Enter Command:")
            print("1. Analyze decision")
            print("2. Reset system") 
            print("3. Exit")
            print("="*50)
            
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == "1":
                decision = input("Enter your decision: ").strip()
                if decision:
                    result = inference.analyze_decision(decision)
                    print("\n" + "="*50)
                    print("ANALYSIS RESULTS:")
                    print(f"Suggestion: {result['suggestion']}")
                    print(f"Regret Risk: {result['regret_risk']}")
                    print(f"Confidence: {result['confidence']}%")
                    print(f"Reasoning: {result['reasoning']}")
                    print(f"Verdict: {result['verdict']}")
                    print("="*50)
                    
            elif choice == "2":
                status = inference.reset()
                print(f"\nReset Status: {status['status']}")
                
            elif choice == "3":
                print("[START] Shutdown initiated")
                print("[STEP] Cleaning up...")
                print("[END] RegretZero Inference Server Stopped")
                break
                
            else:
                print("Invalid choice. Please enter 1-3.")
                
    except KeyboardInterrupt:
        print("\n[START] Interrupt received")
        print("[STEP] Graceful shutdown...")
        print("[END] RegretZero Inference Server Stopped")

if __name__ == "__main__":
    main()
