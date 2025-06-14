from open_webui.models.users import Users, UserModel, UserNameResponse

BOT_USER_ID = "729a12f9-31a5-4123-b6dba-653a53f2c5cf"
BOT_USER_NAME = "AI"
BOT_USER_ROLE = "user"
BOT_USER_PROFILE_IMAGE_URL = "/static/favicon.png"


def get_bot_user_id() -> str:
    return BOT_USER_ID


def get_bot_user() -> UserNameResponse:
    # This should be cached in the future
    user = Users.get_user_by_id(BOT_USER_ID)
    if user:
        return UserNameResponse(**user.model_dump())

    Users.insert_new_user(
        id=BOT_USER_ID,
        name=BOT_USER_NAME,
        email=f"{BOT_USER_ID}@localhost",
        role=BOT_USER_ROLE,
        profile_image_url=BOT_USER_PROFILE_IMAGE_URL,
    )
    user = Users.get_user_by_id(BOT_USER_ID)
    if user:
        return UserNameResponse(**user.model_dump())

    # This should not happen
    raise Exception("Could not create or get bot user") 