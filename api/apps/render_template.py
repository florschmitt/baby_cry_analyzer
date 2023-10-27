from fastapi.templating import Jinja2Templates
from api.core.constants import TEMPLATE_DIR


async def render_template(template_name: str, context: dict):
    templates = Jinja2Templates(directory=TEMPLATE_DIR)
    return templates.TemplateResponse(template_name, context)
