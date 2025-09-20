from __future__ import annotations
import json
import ast
from datetime import datetime
from typing import List, Dict, Any, Optional

from langgraph.types import Command, Interrupt
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, AnyMessage


class ChatHandler:
    
    title = None
    subtitle = None
    events: list[AnyMessage | Interrupt] = []
    new_events: list[AnyMessage | Interrupt] = []
    
    def __init__(self, chat_id=None, title=None, subtitle=None):
        self.chat_id = chat_id
        self.title = title
        self.subtitle = subtitle
        
    def add_events(self, event: AnyMessage | Interrupt | list[AnyMessage | Interrupt]):
        if isinstance(event, list):
            self.events.extend(event)
            self.new_events.extend(event)
        else:
            self.events.append(event)
            self.new_events.append(event)
        

    @property
    def get_new_events(self):
        """Returns the new events and clears the list."""
        new_events = self.new_events.copy()
        self.new_events.clear()
        return new_events
    
    
    def chat2json(self, chat: list[AnyMessage | Interrupt] | None = None) -> list[dict]:
        """
        Convert a chat to a JSON string.
        """
    
        if chat is None:
            chat = self.events
    
        def human_message_to_dict(msg: HumanMessage) -> dict:
            return {
                "role": "user",
                "content": msg.content,
                "resume_interrupt": msg.resume_interrupt if hasattr(msg, 'resume_interrupt') else None,
            }
        
        def ai_message_to_dict(msg: AIMessage) -> dict:
            return {
                "role": "ai",
                "content": msg.content,
                "tool_calls": msg.tool_calls if msg.tool_calls else [],
            }
            
        def tool_message_to_dict(msg: ToolMessage) -> dict:
            return {
                "role": "tool",
                "content": msg.content,
                "name": msg.name,
                "id": msg.id,
                "tool_call_id": msg.tool_call_id
            }
            
        def interrupt_to_dict(msg: Interrupt) -> dict:
            return {
                "role": "interrupt",
                "content": msg.value['content'],
                "interrupt_type": msg.value['interrupt_type'],
                "resumable": msg.resumable,
                "ns": msg.ns
            }
            
        message_type_map = {
            HumanMessage: human_message_to_dict,
            AIMessage: ai_message_to_dict,
            ToolMessage: tool_message_to_dict,
            Interrupt: interrupt_to_dict
        }
        
        chat_dict = [
            message_type_map[type(msg)](msg)
            for msg in chat 
            if type(msg) in message_type_map
        ]
        
        return chat_dict
    
    
    def chat_to_markdown(
        self,
        chat: List[AnyMessage | Interrupt] | None = None,
        path: str | None = None,
        title: str | None = None,
        subtitle: Optional[str] = None,
        include_toc: bool = True,
    ) -> str:
        """
        Genera un file Markdown 'bello' a partire da una lista di messaggi (chat dict).
        - chat: lista di messaggi tipo quelli dell'esempio
        - path: percorso del file .md da creare
        - title/subtitle: titolo opzionale
        - include_toc: inserisce una piccola TOC con ancore ai messaggi
        Ritorna la stringa Markdown generata.
        """
        
        if chat is None:
            chat = self.events
        chat = self.chat2json(chat)
        if not chat:
            return None
        if path is None:
            path = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        if title is None:
            title = self.title if self.title else f"Chat Markdown {datetime.now().isoformat()}"
                    
        
        ROLE_META = {
            "user":      {"emoji": "👤", "label": "User", "color": "#1f6feb"},
            "ai":        {"emoji": "🤖", "label": "Assistant", "color": "#8250df"},
            "tool":      {"emoji": "🛠️", "label": "Tool", "color": "#3fb950"},
            "interrupt": {"emoji": "⏸️", "label": "Interrupt", "color": "#d29922"},
            # fallback handled in code
        }

        def _fence(text: str, lang: str = "") -> str:
            """
            Ritorna un fenced code block che non collida con eventuali triple backtick nel testo.
            Se nel testo compaiono ``` usa ```` come recinzione.
            """
            fence = "```"
            if "```" in text:
                fence = "````"
            lang_tag = lang if lang else ""
            return f"{fence}{lang_tag}\n{text}\n{fence}"

        def _pretty(obj: Any) -> str:
            """Pretty JSON (anche da dict Python), con fallback su str()."""
            try:
                return json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                return str(obj)

        def _maybe_parse_python_like_dict(s: str) -> Any:
            """
            Alcuni tool restituiscono stringhe con apici singoli (non JSON valido).
            Provo a fare ast.literal_eval e poi convertirlo a JSON-friendly.
            """
            try:
                val = ast.literal_eval(s)
                return val
            except Exception:
                # Prova JSON diretto, altrimenti torna la stringa originale
                try:
                    return json.loads(s)
                except Exception:
                    return s
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines: List[str] = []

        # Front matter leggero (opzionale)
        lines.append(f"---")
        lines.append(f'title: "{title}"')
        if subtitle:
            lines.append(f'subtitle: "{subtitle}"')
        lines.append(f"generated: {now}")
        lines.append(f"---\n")

        # Header
        lines.append(f"# {title}")
        if subtitle:
            lines.append(f"_{subtitle}_")
        lines.append(f"*Generato il {now}*")
        lines.append("")
        lines.append("**Legenda**: 👤 User · 🤖 Assistant · 🛠️ Tool · ⏸️ Interrupt")
        lines.append("")

        # TOC
        if include_toc:
            lines.append("## Indice")
            for i, msg in enumerate(chat, 1):
                role = msg.get("role", "unknown")
                meta = ROLE_META.get(role, {"emoji": "📦", "label": role.capitalize()})
                snippet = (msg.get("content") or "").strip().split("\n")[0]
                snippet = snippet if snippet else "(vuoto)"
                # Limita snippet
                if len(snippet) > 80:
                    snippet = snippet[:77] + "…"
                lines.append(f"- [{i:02d} · {meta['emoji']} {meta['label']} – {snippet}](#msg-{i:02d})")
            lines.append("")

        # Messaggi
        for i, msg in enumerate(chat, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            meta = ROLE_META.get(role, {"emoji": "📦", "label": role.capitalize(), "color": "#6e7781"})
            emoji, label = meta["emoji"], meta["label"]

            lines.append(f"---")
            lines.append(f"### {i:02d} · {emoji} {label}")
            lines.append(f"<a id='msg-{i:02d}'></a>")

            # Badge ruoli & extra info
            extras = []
            if role == "interrupt":
                itype = msg.get("interrupt_type") or msg.get("interrupt", {}).get("type")
                if itype:
                    extras.append(f"**Tipo:** `{itype}`")
                resumable = msg.get("resumable")
                if resumable is not None:
                    extras.append(f"**Resumable:** `{resumable}`")
                ns = msg.get("ns")
                if ns:
                    extras.append(f"**Namespaces:** {_fence(_pretty(ns), 'json')}")
            if role == "user" and msg.get("resume_interrupt"):
                extras.append("**Ripresa interrupt:**")
                extras.append(_fence(_pretty(msg["resume_interrupt"]), "json"))

            if extras:
                lines.append("")
                lines.extend(extras)

            # Corpo messaggio
            if content.strip():
                # Se sembra codice Python (inizia con "The generated code is as follows:" o contiene ```python)
                if "```" in content:
                    # già formattato: lo includo così com'è
                    lines.append("")
                    lines.append(content)
                else:
                    # messaggio normale: uso blockquote per user/ai, code fence per tool se è strutturato
                    if role in ("user", "ai"):
                        lines.append("")
                        for ln in content.splitlines():
                            lines.append(f"> {ln}" if ln.strip() else ">")
                    elif role in ("tool", "interrupt"):
                        # spesso i tool mandano json/stringhe strutturate
                        parsed = _maybe_parse_python_like_dict(content)
                        if isinstance(parsed, (dict, list)):
                            lines.append("")
                            lines.append(_fence(_pretty(parsed), "json"))
                        else:
                            lines.append("")
                            lines.append(_fence(str(parsed), ""))
                    else:
                        lines.append("")
                        lines.append(content)
            else:
                # nessun contenuto, ma potrebbero esserci tool_calls
                if role in ("ai", "tool", "interrupt"):
                    pass  # gestito sotto se ci sono tool_calls
                else:
                    lines.append("\n_(nessun contenuto)_")

            # tool_calls (tipicamente dentro messaggi 'ai')
            tcs = msg.get("tool_calls") or []
            if tcs:
                lines.append("")
                lines.append("<details>")
                lines.append("<summary><strong>Tool calls</strong></summary>\n")
                for j, tc in enumerate(tcs, 1):
                    name = tc.get("name") or tc.get("tool")
                    tc_id = tc.get("id") or tc.get("tool_call_id")
                    args = tc.get("args") or {}
                    tc_type = tc.get("type") or ""
                    lines.append(f"**{j}. {name}**  ")
                    if tc_type:
                        lines.append(f"- Tipo: `{tc_type}`")
                    if tc_id:
                        lines.append(f"- ID: `{tc_id}`")
                    if args:
                        lines.append("- Args:")
                        lines.append(_fence(_pretty(args), "json"))
                    # altri campi grezzi, se presenti
                    extra_keys = {k: v for k, v in tc.items() if k not in {"name","args","id","type"}}
                    if extra_keys:
                        lines.append("- Extra:")
                        lines.append(_fence(_pretty(extra_keys), "json"))
                    lines.append("")
                lines.append("</details>")

            lines.append("")

        markdown = "\n".join(lines).rstrip() + "\n"

        # Scrivi su file
        with open(path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return markdown