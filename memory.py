sessions = {}

def get_history(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]


def add_message(session_id: str, role: str, content: str):
    sessions[session_id].append({
        "role": role,
        "content": content
    })
