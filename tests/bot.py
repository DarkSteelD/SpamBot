import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "RUSpam/spam_deberta_v4"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
GROUP_ID = 'your_group_id' 
ACCESS_TOKEN = 'your_access_token'  
THRESHOLD_REACTIONS = 3 

def is_spam(message):
    inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return True if predicted_class == 1 else False

# Основной класс бота
class VkBot:
    def __init__(self, group_id, token):
        self.group_id = group_id
        self.vk_session = vk_api.VkApi(token=token)
        self.vk = self.vk_session.get_api()
        self.longpoll = VkBotLongPoll(self.vk_session, group_id)
        self.message_reactions = {} 
    def run(self):
        print("Бот запущен!")
        for event in self.longpoll.listen():
            if event.type == VkBotEventType.WALL_POST_NEW:
                post_id = event.object["id"]
                user_id = event.object["from_id"]
                text = event.object["text"]

                if is_spam(text):
                    print(f"Обнаружено спам-сообщение: {text}")
                    self.comment_warning(post_id)
                    self.message_reactions[post_id] = {"user_id": user_id, "reactions": 0}

            self.check_reactions()

    def comment_warning(self, post_id):
        """Оставить комментарий под спам-постом."""
        self.vk.wall.createComment(owner_id=-self.group_id, post_id=post_id, message="3 лайка (реакции) и я его забаню.")
        print("Комментарий добавлен!")

    def check_reactions(self):
        """Проверить количество лайков и забанить при необходимости."""
        for post_id, data in list(self.message_reactions.items()):
            likes = self.vk.likes.getList(type="post", owner_id=-self.group_id, item_id=post_id)["count"]
            print(f"Проверка реакций: Пост {post_id} — {likes} лайков.")

            if likes >= THRESHOLD_REACTIONS:
                print(f"Пост {post_id} достиг {likes} реакций. Блокирую пользователя {data['user_id']}!")
                self.ban_user(data["user_id"])
                del self.message_reactions[post_id]

    def ban_user(self, user_id):
        """Забанить пользователя."""
        self.vk.groups.ban(group_id=self.group_id, owner_id=user_id)
        print(f"Пользователь {user_id} забанен.")

# Запуск бота
if __name__ == "__main__":
    bot = VkBot(GROUP_ID, ACCESS_TOKEN)
    bot.run()
