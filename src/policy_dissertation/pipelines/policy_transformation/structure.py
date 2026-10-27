"""Conservative syntax recognition shared by assessment and chunking."""
import re

MODALS = re.compile(r"\b(?:MUST NOT|MUST|SHOULD|MAY|Strictly\s+Prohibited)\b")
NUMBER = re.compile(r"^(\d+(?:\.\d+)*)(?:[.)])?\s+(.+)$")
BULLET = re.compile(r"^(?:[-+*•●▪]|G)\s+")


def heading(text: str, is_code: bool = False) -> tuple[int, str | None, str] | None:
    if is_code:
        return None
    atx = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", text)
    if atx:
        level, title = len(atx[1]), atx[2].replace("**", "").strip()
    else:
        title = text.strip()
        # Plain numbered headings are intentionally narrow; sentences and rules stay prose.
        if not NUMBER.match(title) or not NUMBER.match(title)[2][0].isupper() or len(title) > 120 or MODALS.search(title) or title.endswith(('.', ';', ':')):
            return None
        level = len(NUMBER.match(title)[1].split('.')) + 1
    numbered = NUMBER.match(title)
    return level, numbered[1] if numbered else None, numbered[2] if numbered else title


def is_table(text: str) -> bool:
    return text.lstrip().startswith('|')


def table_cells(text: str) -> list[str]:
    return [cell.strip() for cell in re.split(r'(?<!\\)\|', text.strip().strip('|'))]


def is_table_separator(text: str) -> bool:
    return all(re.fullmatch(r':?-{3,}:?', cell) for cell in table_cells(text))
