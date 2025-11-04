import boto3
import json

def test_claude_sonnet():
    # Create Bedrock runtime client
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="ap-east-1")

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    prompt = "Write a short 2-line poem about AWS and AI."

    # ✅ Correct request format for Anthropic models on Bedrock
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    })

    print("🔹 Sending request to Claude 3.5 Sonnet on Bedrock...")
    response = bedrock.invoke_model(
        modelId=model_id,
        body=body
    )

    result = json.loads(response["body"].read())
    print("✅ Claude Response:\n", result["content"][0]["text"])

if __name__ == "__main__":
    test_claude_sonnet()
