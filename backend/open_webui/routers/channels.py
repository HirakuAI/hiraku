import json
import logging
import re
import asyncio
from typing import Optional


from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks
from pydantic import BaseModel


from open_webui.socket.main import sio, get_user_ids_from_room
from open_webui.models.users import Users, UserModel, UserNameResponse

from open_webui.models.channels import Channels, ChannelModel, ChannelForm
from open_webui.models.messages import (
    Messages,
    MessageModel,
    MessageResponse,
    MessageForm,
)


from open_webui.config import ENABLE_ADMIN_CHAT_ACCESS, ENABLE_ADMIN_EXPORT
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS


from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access, get_users_with_access
from open_webui.utils.webhook import post_webhook
from open_webui.utils.bot import get_bot_user_id, get_bot_user
from open_webui.routers.openai import (
    generate_chat_completion as generate_openai_chat_completion,
)
from open_webui.routers.ollama import (
    generate_chat_completion as generate_ollama_chat_completion,
)
from open_webui.utils.models import get_all_models

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()

# Helper function for ERROR_MESSAGES since they are constants, not callables
def _err(name="DEFAULT"):
    return ERROR_MESSAGES[name]

############################
# GetChatList
############################


@router.get("/", response_model=list[ChannelModel])
async def get_channels(user=Depends(get_verified_user)):
    if user.role == "admin":
        return Channels.get_channels()
    else:
        return Channels.get_channels_by_user_id(user.id)


############################
# CreateNewChannel
############################


@router.post("/create", response_model=Optional[ChannelModel])
async def create_new_channel(form_data: ChannelForm, user=Depends(get_admin_user)):
    try:
        channel = Channels.insert_new_channel(None, form_data, user.id)
        if channel:
            return ChannelModel(**channel.model_dump())
        return None
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )


############################
# GetChannelById
############################


@router.get("/{id}", response_model=Optional[ChannelModel])
async def get_channel_by_id(id: str, user=Depends(get_verified_user)):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="read", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    return ChannelModel(**channel.model_dump())


############################
# UpdateChannelById
############################


@router.post("/{id}/update", response_model=Optional[ChannelModel])
async def update_channel_by_id(
    id: str, form_data: ChannelForm, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="write", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    try:
        updated_channel = Channels.update_channel_by_id(id, form_data)
        if updated_channel:
            return ChannelModel(**updated_channel.model_dump())
        return None
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )


############################
# DeleteChannelById
############################


@router.delete("/{id}/delete", response_model=bool)
async def delete_channel_by_id(id: str, user=Depends(get_admin_user)):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    try:
        Channels.delete_channel_by_id(id)
        return True
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )


############################
# GetChannelMessages
############################


class MessageUserResponse(MessageResponse):
    user: UserNameResponse


@router.get("/{id}/messages", response_model=list[MessageUserResponse])
async def get_channel_messages(
    id: str, skip: int = 0, limit: int = 50, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="read", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    message_list = Messages.get_messages_by_channel_id(id, skip, limit)
    users = {}

    messages = []
    for message in message_list:
        message_user = Users.get_user_by_id(message.user_id)
        if message_user:
            users[message.user_id] = message_user

        replies = Messages.get_replies_by_message_id(message.id)
        latest_reply_at = replies[0].created_at if replies else None

        messages.append(
            MessageUserResponse(
                **{
                    **message.model_dump(),
                    "reply_count": len(replies),
                    "latest_reply_at": latest_reply_at,
                    "reactions": Messages.get_reactions_by_message_id(message.id),
                    "user": UserNameResponse(**users[message.user_id].model_dump()),
                }
            )
        )

    return messages


############################
# GetChannelUsers
############################


@router.get("/{id}/users", response_model=list[UserNameResponse])
async def get_channel_users(id: str, user=Depends(get_verified_user)):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="read", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    users_raw = get_users_with_access("read", channel.access_control)
    users = []
    for u in users_raw:
        # u is either UserModel or a plain str (user-id)
        if isinstance(u, str):
            user_obj = Users.get_user_by_id(u)
            if user_obj:
                users.append(UserNameResponse(**user_obj.model_dump()))
        else:
            users.append(UserNameResponse(**u.model_dump()))
    return users


############################
# PostNewMessage
############################


async def send_notification(name, webui_url, channel, message, active_user_ids):
    users = get_users_with_access("read", channel.access_control)

    for user in users:
        if user.id in active_user_ids:
            continue
        else:
            if user.settings:
                webhook_url = user.settings.ui.get("notifications", {}).get(
                    "webhook_url", None
                )

                if webhook_url:
                    post_webhook(
                        name,
                        webhook_url,
                        f"#{channel.name} - {webui_url}/channels/{channel.id}\n\n{message.content}",
                        {
                            "action": "channel",
                            "message": message.content,
                            "title": channel.name,
                            "url": f"{webui_url}/channels/{channel.id}",
                        },
                    )


async def handle_bot_mention(
    request: Request, channel: ChannelModel, message: MessageModel, user: UserModel
):
    if not channel.bot_enabled or not channel.bot_name or not channel.bot_model:
        return
        
    if f"@{channel.bot_name}" not in message.content:
        return
        
    if not request.app.state.MODELS:
        await get_all_models(request, user=user)

    log.info(f"Handling bot mention in channel {channel.id}")

    # Get message history
    history = Messages.get_messages_by_channel_id(channel.id)
    messages_for_bot = [
        {
            "role": "assistant" if h.user_id == get_bot_user_id() else "user",
            "content": h.content,
        }
        for h in reversed(history)
    ]

    # Add the current message to the history
    messages_for_bot.append({"role": "user", "content": message.content})

    try:
        bot_user = get_bot_user()

        # Emit typing indicator
        await sio.emit(
            "channel-events",
            {
                "channel_id": channel.id,
                "data": {"type": "typing", "data": {"typing": True}},
                "user": bot_user.model_dump(),
            },
            to=f"channel:{channel.id}",
        )

        form_data = {
            "model": channel.bot_model,
            "messages": messages_for_bot,
            "stream": True,
            **(channel.bot_config if channel.bot_config else {}),
        }

        # Determine if the model is from Ollama or OpenAI
        model_info = request.app.state.MODELS.get(channel.bot_model, {})
        is_ollama_model = model_info.get("owned_by") == "ollama"

        if is_ollama_model:
            response = await generate_ollama_chat_completion(request, form_data, user)
        else:
            response = await generate_openai_chat_completion(request, form_data, user)

        response_content = ""
        async for chunk in response.body_iterator:
            try:
                if isinstance(chunk, bytes):
                    chunk_str = chunk.decode("utf-8")
                    if chunk_str.startswith("data: "):
                        chunk_str = chunk_str[6:]
                    if chunk_str.strip() == "[DONE]":
                        continue
                    data = json.loads(chunk_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            response_content += content
            except json.JSONDecodeError:
                log.warning(f"Failed to decode JSON chunk: {chunk}")
            except Exception as e:
                log.error(f"Error processing stream chunk: {e}")

        # Stop typing indicator
        await sio.emit(
            "channel-events",
            {
                "channel_id": channel.id,
                "data": {"type": "typing", "data": {"typing": False}},
                "user": bot_user.model_dump(),
            },
            to=f"channel:{channel.id}",
        )

        if response_content:
            bot_message_form = MessageForm(content=response_content)
            bot_message = Messages.insert_new_message(
                bot_message_form, channel.id, get_bot_user_id()
            )
            
            if bot_message:
                event_data = {
                    "channel_id": channel.id,
                    "message_id": bot_message.id,
                    "data": {
                        "type": "message",
                        "data": MessageUserResponse(
                            **{
                                **bot_message.model_dump(),
                                "reply_count": 0,
                                "latest_reply_at": None,
                                "reactions": [],
                                "user": bot_user,
                            }
                        ).model_dump(),
                    },
                    "user": bot_user.model_dump(),
                    "channel": channel.model_dump(),
                }
                
                await sio.emit(
                    "channel-events",
                    event_data,
                    to=f"channel:{channel.id}",
                )
                
    except Exception as e:
        log.error(f"Error handling bot mention: {e}")
        error_message = f"Sorry, I encountered an error: {e}"
        bot_message_form = MessageForm(content=error_message)
        error_msg_db = Messages.insert_new_message(
            bot_message_form, channel.id, get_bot_user_id()
        )
        if error_msg_db:
            await sio.emit(
                "channel-events",
                {
                    "channel_id": channel.id,
                    "message_id": error_msg_db.id,
                    "data": {
                        "type": "message",
                        "data": MessageUserResponse(
                            **{
                                **error_msg_db.model_dump(),
                                "reply_count": 0,
                                "latest_reply_at": None,
                                "reactions": [],
                                "user": get_bot_user(),
                            }
                        ).model_dump(),
                    },
                    "user": get_bot_user().model_dump(),
                    "channel": channel.model_dump(),
                },
                to=f"channel:{channel.id}",
            )


@router.post("/{id}/messages/post", response_model=Optional[MessageModel])
async def post_new_message(
    request: Request,
    id: str,
    form_data: MessageForm,
    background_tasks: BackgroundTasks,
    user=Depends(get_verified_user),
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="read", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    try:
        message = Messages.insert_new_message(form_data, channel.id, user.id)

        if message:
            event_data = {
                "channel_id": channel.id,
                "message_id": message.id,
                "data": {
                    "type": "message",
                    "data": MessageUserResponse(
                        **{
                            **message.model_dump(),
                            "reply_count": 0,
                            "latest_reply_at": None,
                            "reactions": Messages.get_reactions_by_message_id(
                                message.id
                            ),
                            "user": UserNameResponse(**user.model_dump()),
                        }
                    ).model_dump(),
                },
                "user": UserNameResponse(**user.model_dump()).model_dump(),
                "channel": channel.model_dump(),
            }

            await sio.emit(
                "channel-events",
                event_data,
                to=f"channel:{channel.id}",
            )

            if message.parent_id:
                # If this message is a reply, emit to the parent message as well
                parent_message = Messages.get_message_by_id(message.parent_id)

                if parent_message:
                    parent_message_user = Users.get_user_by_id(parent_message.user_id)
                    if parent_message_user:
                        await sio.emit(
                            "channel-events",
                            {
                                "channel_id": channel.id,
                                "message_id": parent_message.id,
                                "data": {
                                    "type": "message:reply",
                                    "data": MessageUserResponse(
                                        **{
                                            **parent_message.model_dump(),
                                            "user": UserNameResponse(
                                                **parent_message_user.model_dump()
                                            ),
                                        }
                                    ).model_dump(),
                                },
                                "user": UserNameResponse(**user.model_dump()).model_dump(),
                                "channel": channel.model_dump(),
                            },
                            to=f"channel:{channel.id}",
                        )

            active_user_ids = get_user_ids_from_room(f"channel:{channel.id}")

            background_tasks.add_task(
                send_notification,
                request.app.state.WEBUI_NAME,
                request.app.state.config.WEBUI_URL,
                channel,
                message,
                active_user_ids,
            )

            background_tasks.add_task(
                handle_bot_mention, request, channel, message, user
            )

            return MessageModel(**message.model_dump())
        return None
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )


############################
# GetChannelMessage
############################


@router.get("/{id}/messages/{message_id}", response_model=Optional[MessageUserResponse])
async def get_channel_message(
    id: str, message_id: str, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="read", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )

    message_user = Users.get_user_by_id(message.user_id)
    if not message_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("USER_NOT_FOUND")
        )

    return MessageUserResponse(
        **{
            **message.model_dump(),
            "user": UserNameResponse(**message_user.model_dump()),
        }
    )


############################
# GetChannelThreadMessages
############################


@router.get(
    "/{id}/messages/{message_id}/thread", response_model=list[MessageUserResponse]
)
async def get_channel_thread_messages(
    id: str,
    message_id: str,
    skip: int = 0,
    limit: int = 50,
    user=Depends(get_verified_user),
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="read", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    message_list = Messages.get_messages_by_parent_id(id, message_id, skip, limit)
    users = {}

    messages = []
    for message in message_list:
        message_user = Users.get_user_by_id(message.user_id)
        if message_user:
            users[message.user_id] = message_user

        messages.append(
            MessageUserResponse(
                **{
                    **message.model_dump(),
                    "reply_count": 0,
                    "latest_reply_at": None,
                    "reactions": Messages.get_reactions_by_message_id(message.id),
                    "user": UserNameResponse(**users[message.user_id].model_dump()),
                }
            )
        )

    return messages


############################
# UpdateMessageById
############################


@router.post(
    "/{id}/messages/{message_id}/update", response_model=Optional[MessageModel]
)
async def update_message_by_id(
    id: str, message_id: str, form_data: MessageForm, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="read", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )

    try:
        updated_message = Messages.update_message_by_id(message_id, form_data)

        if updated_message:
            message = Messages.get_message_by_id(message_id)
            if message:
                await sio.emit(
                    "channel-events",
                    {
                        "channel_id": channel.id,
                        "message_id": message.id,
                        "data": {
                            "type": "message:update",
                            "data": MessageUserResponse(
                                **{
                                    **message.model_dump(),
                                    "user": UserNameResponse(
                                        **user.model_dump()
                                    ).model_dump(),
                                }
                            ).model_dump(),
                        },
                        "user": UserNameResponse(**user.model_dump()).model_dump(),
                        "channel": channel.model_dump(),
                    },
                    to=f"channel:{channel.id}",
                )

                return MessageModel(**message.model_dump())
        return None
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )


############################
# AddReactionToMessage
############################


class ReactionForm(BaseModel):
    name: str


@router.post("/{id}/messages/{message_id}/reactions/add", response_model=bool)
async def add_reaction_to_message(
    id: str, message_id: str, form_data: ReactionForm, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="read", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )

    try:
        Messages.add_reaction_to_message(message_id, user.id, form_data.name)
        message = Messages.get_message_by_id(message_id)

        if message:
            message_user = Users.get_user_by_id(message.user_id)
            if message_user:
                await sio.emit(
                    "channel-events",
                    {
                        "channel_id": channel.id,
                        "message_id": message.id,
                        "data": {
                            "type": "message:reaction:add",
                            "data": {
                                **message.model_dump(),
                                "user": UserNameResponse(
                                    **message_user.model_dump()
                                ).model_dump(),
                                "name": form_data.name,
                            },
                        },
                        "user": UserNameResponse(**user.model_dump()).model_dump(),
                        "channel": channel.model_dump(),
                    },
                    to=f"channel:{channel.id}",
                )

        return True
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )


############################
# RemoveReactionById
############################


@router.post("/{id}/messages/{message_id}/reactions/remove", response_model=bool)
async def remove_reaction_by_id_and_user_id_and_name(
    id: str, message_id: str, form_data: ReactionForm, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="read", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )

    try:
        Messages.remove_reaction_by_id_and_user_id_and_name(
            message_id, user.id, form_data.name
        )

        message = Messages.get_message_by_id(message_id)
        if message:
            message_user = Users.get_user_by_id(message.user_id)
            if message_user:
                await sio.emit(
                    "channel-events",
                    {
                        "channel_id": channel.id,
                        "message_id": message.id,
                        "data": {
                            "type": "message:reaction:remove",
                            "data": {
                                **message.model_dump(),
                                "user": UserNameResponse(
                                    **message_user.model_dump()
                                ).model_dump(),
                                "name": form_data.name,
                            },
                        },
                        "user": UserNameResponse(**user.model_dump()).model_dump(),
                        "channel": channel.model_dump(),
                    },
                    to=f"channel:{channel.id}",
                )

        return True
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )


############################
# DeleteMessageById
############################


@router.delete("/{id}/messages/{message_id}/delete", response_model=bool)
async def delete_message_by_id(
    id: str, message_id: str, user=Depends(get_verified_user)
):
    channel = Channels.get_channel_by_id(id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if user.role != "admin" and not has_access(
        user.id, type="read", access_control=channel.access_control
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_err("ACCESS_PROHIBITED"),
        )

    message = Messages.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_err("NOT_FOUND")
        )

    if message.channel_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )

    try:
        Messages.delete_message_by_id(message_id)
        await sio.emit(
            "channel-events",
            {
                "channel_id": channel.id,
                "message_id": message.id,
                "data": {
                    "type": "message:delete",
                    "data": {
                        **message.model_dump(),
                        "user": UserNameResponse(**user.model_dump()).model_dump(),
                    },
                },
                "user": UserNameResponse(**user.model_dump()).model_dump(),
                "channel": channel.model_dump(),
            },
            to=f"channel:{channel.id}",
        )

        if message.parent_id:
            parent_message = Messages.get_message_by_id(message.parent_id)

            if parent_message:
                parent_message_user = Users.get_user_by_id(parent_message.user_id)
                if parent_message_user:
                    await sio.emit(
                        "channel-events",
                        {
                            "channel_id": channel.id,
                            "message_id": parent_message.id,
                            "data": {
                                "type": "message:reply",
                                "data": MessageUserResponse(
                                    **{
                                        **parent_message.model_dump(),
                                        "user": UserNameResponse(
                                            **parent_message_user.model_dump()
                                        ),
                                    }
                                ).model_dump(),
                            },
                            "user": UserNameResponse(**user.model_dump()).model_dump(),
                            "channel": channel.model_dump(),
                        },
                        to=f"channel:{channel.id}",
                    )

        return True
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_err("DEFAULT")
        )


# Unit test sentinel (optional)
if __name__ == "__main__":
    # Quick sanity check; run `python -m open_webui.routers.channels`
    from open_webui.models.users import Users
    assert isinstance(_err(), str)
    assert Users.get_user_by_id(get_bot_user_id())  # bot exists
    print("router imports OK")
