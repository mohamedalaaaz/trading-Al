"""
Bitcoin & Crypto News Fetcher
Uses Anthropic API with web search to get the latest important news.
Requirements: pip install anthropic
Set your API key: export ANTHROPIC_API_KEY="your_key_here"
"""

import anthropic
import json
import os
from datetime import datetime


def get_news(topic: str = "Bitcoin", count: int = 5) -> list[dict]:
    """Fetch latest important news for a given topic."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""Search the web and find the {count} most important and recent news stories about {topic} from today or this week.

Return ONLY a valid JSON array with no markdown, no backticks, no extra text. Use this exact structure:
[
  {{
    "title": "News headline here",
    "summary": "One or two sentence summary of the story.",
    "category": "One of: Price, Markets, Regulation, Mining, Macro, ETF, Adoption, Technology, General",
    "time": "e.g. 2 hours ago / Today / Yesterday",
    "importance": "High / Medium / Low"
  }}
]"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text from response
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    # Clean and parse JSON
    text = text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def print_news(news_items: list[dict], topic: str):
    """Pretty print the news items in the terminal."""
    width = 70
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("\n" + "=" * width)
    print(f"  📰  {topic.upper()} NEWS FEED  |  {now}")
    print("=" * width)

    importance_icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    category_icons = {
        "Price": "💰", "Markets": "📈", "Regulation": "⚖️",
        "Mining": "⛏️", "Macro": "🌍", "ETF": "📊",
        "Adoption": "🤝", "Technology": "💻", "General": "📰"
    }

    for i, item in enumerate(news_items, 1):
        importance = item.get("importance", "Medium")
        category = item.get("category", "General")
        icon = importance_icons.get(importance, "🟡")
        cat_icon = category_icons.get(category, "📰")

        print(f"\n  {i}. {icon} [{category}] {cat_icon}  —  {item.get('time', '')}")
        print(f"     {item['title']}")
        print(f"     {item['summary']}")
        print("  " + "-" * (width - 2))

    print(f"\n  Total: {len(news_items)} stories fetched\n")


def save_to_file(news_items: list[dict], topic: str, filename: str = None):
    """Save news to a JSON file."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"news_{topic.lower().replace(' ', '_')}_{timestamp}.json"

    output = {
        "topic": topic,
        "fetched_at": datetime.now().isoformat(),
        "count": len(news_items),
        "stories": news_items
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  💾  Saved to: {filename}")
    return filename


def main():
    # ── CONFIGURE HERE ──────────────────────────────────────────────
    TOPIC = "Bitcoin"          # Change to any topic: "Tesla", "AI", "Gold", etc.
    COUNT = 5                  # Number of news stories to fetch
    SAVE_JSON = True           # Set to False to skip saving
    # ────────────────────────────────────────────────────────────────

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n❌  Error: ANTHROPIC_API_KEY not set.")
        print("   Run:  export ANTHROPIC_API_KEY='your_key_here'\n")
        return

    print(f"\n🔍  Searching for latest {TOPIC} news...")

    try:
        news = get_news(topic=TOPIC, count=COUNT)
        print_news(news, topic=TOPIC)

        if SAVE_JSON:
            save_to_file(news, topic=TOPIC)

    except json.JSONDecodeError as e:
        print(f"\n❌  Failed to parse news response: {e}")
    except anthropic.AuthenticationError:
        print("\n❌  Invalid API key. Check your ANTHROPIC_API_KEY.")
    except Exception as e:
        print(f"\n❌  Error: {e}")


if __name__ == "__main__":
    main()
