import json
import time
import uuid
from typing import Optional

from open_webui.internal.db import Base, get_db
from open_webui.utils.access_control import has_access

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Boolean, Column, String, Text, JSON
from sqlalchemy import or_, func, select, and_, text
from sqlalchemy.sql import exists

####################
# Channel DB Schema
####################


class Channel(Base):
    __tablename__ = "channel"

    id = Column(Text, primary_key=True)
    user_id = Column(Text)
    type = Column(Text, nullable=True)

    name = Column(Text)
    description = Column(Text, nullable=True)

    data = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)
    access_control = Column(JSON, nullable=True)

    bot_name = Column(Text, nullable=True)
    bot_model = Column(Text, nullable=True)
    bot_enabled = Column(Boolean, default=False, nullable=False)
    bot_config = Column(JSON, nullable=True)

    created_at = Column(BigInteger)
    updated_at = Column(BigInteger)


class ChannelModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    type: Optional[str] = None

    name: str
    description: Optional[str] = None

    data: Optional[dict] = None
    meta: Optional[dict] = None
    access_control: Optional[dict] = None

    bot_name: Optional[str] = None
    bot_model: Optional[str] = None
    bot_enabled: bool = False
    bot_config: Optional[dict] = None

    created_at: int  # timestamp in epoch
    updated_at: int  # timestamp in epoch


####################
# Forms
####################


class ChannelForm(BaseModel):
    name: str
    description: Optional[str] = None
    data: Optional[dict] = None
    meta: Optional[dict] = None
    access_control: Optional[dict] = None
    bot_name: Optional[str] = None
    bot_model: Optional[str] = None
    bot_enabled: bool = False
    bot_config: Optional[dict] = None


class ChannelTable:
    def insert_new_channel(
        self, type: Optional[str], form_data: ChannelForm, user_id: str
    ) -> Optional[ChannelModel]:
        print(f"DEBUG: insert_new_channel form_data: {form_data}")
        print(f"DEBUG: form_data.bot_enabled: {form_data.bot_enabled}")
        print(f"DEBUG: form_data.bot_name: {form_data.bot_name}")
        print(f"DEBUG: form_data.bot_model: {form_data.bot_model}")
        
        with get_db() as db:
            channel_data = {
                **form_data.model_dump(),
                "type": type,
                "name": form_data.name.lower(),
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "created_at": int(time.time_ns()),
                "updated_at": int(time.time_ns()),
            }
            print(f"DEBUG: channel_data before ChannelModel: {channel_data}")
            
            channel = ChannelModel(**channel_data)
            print(f"DEBUG: ChannelModel created with bot_enabled: {channel.bot_enabled}, bot_name: {channel.bot_name}, bot_model: {channel.bot_model}")

            new_channel = Channel(**channel.model_dump())
            print(f"DEBUG: Channel DB object created with bot_enabled: {new_channel.bot_enabled}, bot_name: {new_channel.bot_name}, bot_model: {new_channel.bot_model}")

            db.add(new_channel)
            db.commit()
            print(f"DEBUG: Channel committed to DB")
            return channel

    def get_channels(self) -> list[ChannelModel]:
        with get_db() as db:
            channels = db.query(Channel).all()
            return [ChannelModel.model_validate(channel) for channel in channels]

    def get_channels_by_user_id(
        self, user_id: str, permission: str = "read"
    ) -> list[ChannelModel]:
        channels = self.get_channels()
        return [
            channel
            for channel in channels
            if channel.user_id == user_id
            or has_access(user_id, permission, channel.access_control)
        ]

    def get_channel_by_id(self, id: str) -> Optional[ChannelModel]:
        with get_db() as db:
            channel = db.query(Channel).filter(Channel.id == id).first()
            if channel:
                print(f"DEBUG: DB channel found - bot_enabled: {channel.bot_enabled}, bot_name: {channel.bot_name}, bot_model: {channel.bot_model}")
                result = ChannelModel.model_validate(channel)
                print(f"DEBUG: ChannelModel after validation - bot_enabled: {result.bot_enabled}, bot_name: {result.bot_name}, bot_model: {result.bot_model}")
                return result
            else:
                print(f"DEBUG: No channel found with id: {id}")
                return None

    def update_channel_by_id(
        self, id: str, form_data: ChannelForm
    ) -> Optional[ChannelModel]:
        with get_db() as db:
            channel = db.query(Channel).filter(Channel.id == id).first()
            if not channel:
                return None

            channel.name = form_data.name
            channel.data = form_data.data
            channel.meta = form_data.meta
            channel.access_control = form_data.access_control
            channel.bot_name = form_data.bot_name
            channel.bot_model = form_data.bot_model
            channel.bot_enabled = form_data.bot_enabled
            channel.bot_config = form_data.bot_config
            channel.updated_at = int(time.time_ns())

            db.commit()
            return ChannelModel.model_validate(channel) if channel else None

    def delete_channel_by_id(self, id: str):
        with get_db() as db:
            db.query(Channel).filter(Channel.id == id).delete()
            db.commit()
            return True


Channels = ChannelTable()
