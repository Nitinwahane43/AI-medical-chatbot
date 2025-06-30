from flask import Flask, request, render_template, session
from generate_responce import get_response

app = Flask(__name__)
app.secret_key = "secure-key"

@app.before_request
def clear_history_on_restart():
    # Reset history when session is first created (fresh browser visit or restart)
    if "initialized" not in session:
        session["chat_history"] = []
        session["initialized"] = True

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            response = get_response(question, session["chat_history"])
            answer = response.get("answer", "⚠️ No answer received.")
            session["chat_history"].append((question, answer))
            session.modified = True

    return render_template("index.html", answer=answer, chat_history=session.get("chat_history", []))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
