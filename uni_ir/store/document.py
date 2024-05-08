from datetime import datetime
from uuid import UUID
from pydantic import Field, BaseModel
from .base import Entity


class Metadata(BaseModel):
    mimetype: str = Field(..., title="MIME type of the document")
    uri: str = Field(..., title="URI of the document")
    title: str | None = Field(default=None, title="Title of the document")
    section: str | None = Field(default=None, title="Section of the document")


class Document(Entity):
    collection_id: UUID | None = None
    content: str = Field(..., title="Content of the document")
    metadata: Metadata = Field(..., title="Metadata of the document")


class Collection(Entity):
    name: str = Field(..., title="Name of the collection")
    description: str | None = Field(default=None, title="Description of the collection")
    created_at: datetime = Field(default_factory=datetime.now, title="Date of creation")
    updated_at: datetime = Field(
        default_factory=datetime.now, title="Date of last update"
    )
