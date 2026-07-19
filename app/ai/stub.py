def get_ai_response(user_message: str, conversation_history: list) -> str:
    history_length = len(conversation_history)
    if history_length <= 1:
        return "Thank you for sharing that. Can you tell me more about how that made you feel?"
    return f"I hear you. We've been talking for {history_length} messages now. What feels most important to explore?"


# if __name__ == "__main__":
#     print(
#         get_ai_response(
#             "I am sad",
#             []
#         )
#     )

