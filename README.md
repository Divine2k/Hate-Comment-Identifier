# Comment Analysis and Policy Check

This project is a simple yet powerful tool designed to analyze and summarize online comments while checking them against a predefined policy. The goal is to help identify whether a comment violates community guidelines and to provide a concise summary of its intent.

## 🚀 Features

- **Multi-language Support**: Detects the language of comments (e.g., English, Hindi) and processes them accordingly.
- **Policy Compliance Check**: Uses Retrieval-Augmented Generation (RAG) to reference policy documents and verify if the comment violates any guidelines.
- **Comment Summarization**: Regardless of policy compliance, the system summarizes the intent and tone of the initial comment.
- **Command-Line Interface (CLI)**: Simple to use via the terminal—no frontend required for now.

---

## 🔍 How It Works

1. **Input**:\
   You provide two messages:

   - `previous_comment`: The original comment or message.
   - `initial_comment`: The new message that responds to the previous one.

   Example:

   ```python
   {
       'previous_comment': 'Will you marry me?',
       'initial_comment': "Yes I'll marry you babe!"
   }
   ```

2. **Text Cleaning & Language Detection**:\
   Both comments are cleaned and checked for language (e.g., English, Hindi). Only supported languages proceed.

3. **Policy Compliance Analysis**:\
   The `initial_comment` is checked against community policies using a Large Language Model (LLM) and RAG to ensure adherence.

4. **Summarization**:\
   Regardless of policy violation, the system uses another LLM to summarize the intent, tone, and meaning of the `initial_comment`.

5. **Output Example**:

   ```python
   {
       'previous_comment': 'Will you marry me?',
       'previous_comment_lang': 'English',
       'initial_comment': "Yes I'll marry you babe!",
       'initial_comment_lang': 'English',
       'initial_comment_analysis_summary': 'Policy Not Violated. The comment is in English, which is not specifically mentioned in the policies.',
       'overall_summary': 'The initial_comment is expressing acceptance of a proposal. It indicates a positive response to a question about marriage. The tone is affectionate, suggesting a romantic relationship. The speaker is agreeing to marry the person who made the proposal. The comment is enthusiastic and loving.'
   }
   ```

---

## 🛠 Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Set API Keys**:\
   You'll need API keys for your LLM provider (e.g., OpenAI, Azure, etc.). Create a `.env` file or set them directly in your script.

3. **Run the Script**:

   ```bash
   python comment_analysis.py
   ```

   Follow the prompts to input your comments.

---

## 📝 Notes

- Currently, the system uses a simple Python dictionary to store and pass the comments. This can easily be replaced with an API call or database integration in future versions.
- There is no frontend yet—this is a command-line-only tool.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas, find bugs, or want to improve the system:

- Fork the repo
- Open a pull request
- Or just drop a comment 🙂

---

## 📄 License

MIT License.

---

👉 **Next Steps (Suggestions)**:

- Add support for more languages.
- Build a web-based interface.
- Improve the policy knowledge base for more nuanced analysis.

