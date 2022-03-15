import telebot
import requests
import torch
import model1
import model2
import music_to_spectograms
from random import randint

API_TOKEN = "" # Вставьте id бота от Father bot
my_bot = telebot.TeleBot(API_TOKEN, parse_mode=None)
main_chat_id = 715687628 # тут нужен ваш id. Его можно получить у этого бота: @getmyid_bot
need_spam = False
last_meow = dict()

first_model = model1.CNNModel()
second_model = model2.Model()
first_model.load_state_dict(torch.load('Weights_model/model1.pth', map_location=torch.device('cpu')))
second_model.load_state_dict(torch.load('Weights_model/model2.pth'))

# Список обычных аниме девочек, их дерикторию и описание
common_anime_girl_meow = [
    ['meow/Hungry_meow.jpg', '*голодное мяу'],
    ['meow/Playful_meow.jpg', '*игривое мяу'],
    ['meow/Shy_meow.jpg', '*стеснительное мяу'],
    ['meow/Surprised_meow.jpg', '*удивленное мяу'],
    ['meow/sXTt0XrNS9U.jpg', '*приветственное мяу']
]

# Список редких аниме девочек, их, описание и вероятность (Если вы впишете число x, то вероятность будет ~ 1 / x)
rare_anime_girl_meow = [
    ['meow/Rare/Angry_anime_meow.jpg', 23, '*супер злобное мяу'],
    ['meow/Rare/Trap_meow.png', 47, 'Oh, no, it`s trap мяу'],
    ['meow/Rare/tinkoff_meow.jpg', 57, 'Мяу от детей Tinkoff Degeneration'],
    ['meow/Rare/Im_buried.png', 69, 'Я зарылся^^'],
    ['meow/Rare/Ban_meow.jpg', 666, '*неполиткоректное мяу'],
    ['meow/Rare/Dead_inside_meow.jpg', 99, 'Dead inside Мяу'],
    ['meow/Rare/Study_meow.jpg', 99, 'Мяу Мяу'],
    ['meow/Rare/Bye_meow.jpg', 17, 'Прощальное мяу('],
    ['meow/Angry_babin_meow.jpg', 15, '*злобное мяу']
]
# Аниме девочки заканчиваются(((((((((


# названия жанров и описания их.
number_to_genre = [
    ['International', 'Internatinal???? А что это додумайте сами'],
    ['Pop', 'Вы прислали POP'],
    ['Rock', 'ЕЕЕЕЕЕ, роцк'],
    ['Electronic', 'ТЫЦ, ТЫЦ, Электроника'],
    ['Folk', 'Это Народные песни'],
    ['Hip-Hop', 'ЁУ, это Хип-хоп'],
    ['Experimental', 'Experimental: ДА!'],
    ['Instrumental', 'Вы прислали инструментальную музыку']
]


# Загружает файл и возвращает его директорию
def save_file_on_pc(file_name, file_url):
    tmp = requests.get(file_url)
    with open('download_files/' + file_name, 'wb') as fd:  
        for chunk in tmp.iter_content(chunk_size=128):  
            fd.write(chunk) 
    print(file_name, 'download!')
    return 'download_files/' + file_name


#Шлет выбранное мяу мяу.
def send_meow(file_path, text_caption, chat_id):
    meow = open(file_path, 'rb')
    my_bot.send_photo(chat_id, meow, caption=text_caption)


#Отправляет всем людям спам.
def send_spam(file_id, caption):
    f = open("proshmandovki_siriusa.txt", 'r')
    all_chats = set()
    for i in f:
        all_chats.add(i.split(':')[0])
    for i in all_chats:
        print(i)
    for i in all_chats:
        try:
            my_bot.send_photo(i, file_id, caption=caption)
        except:
            pass
    print('Спам успешно отправлен!')
    f.close()


#По спектограмме определяет жанр
def spectograms_to_ganre(file_path):
    spectograms = music_to_spectograms.track_to_spectrograms(file_path)
    genre = first_model.forward((torch.from_numpy(spectograms[0]))[None,:])
    cnt = 1
    for i in range(1, len(spectograms)):
        cnt += 2
        genre += first_model.forward((torch.from_numpy(spectograms[0]))[None,:])
        genre += second_model.forward((torch.from_numpy(spectograms[0]))[None,:])
    genre = torch.nn.functional.softmax(genre / cnt, dim=-1)
    return genre


def tensor_to_genre(genre):
    return genre.argmax()


@my_bot.message_handler(commands=['start_spam'])
def start_spam(message):
    global need_spam
    if (message.chat.id == main_chat_id):
        my_bot.reply_to(message, 'Приветсвую тебя, мой господин!')
        need_spam = True
    else:
        my_bot.reply_to(message, 'Ты не мой господин :/')


@my_bot.message_handler(commands=['stop_spam'])
def start_spam(message):
    global need_spam
    if (message.chat.id == main_chat_id):
        my_bot.reply_to(message, 'Приветсвую тебя, мой господин!')
        need_spam = False
    else:
        my_bot.reply_to(message, 'Ты не мой господин :/')


@my_bot.message_handler(commands=['start'])
def command_start(message):
    f = open("proshmandovki_siriusa.txt", 'a')
    f.write(str(message.chat.id) + ':' + message.from_user.username + '\n')
    f.close()
    my_bot.reply_to(message, "Хайп! Для получения более подробной информации напишите /help")


@my_bot.message_handler(commands=['help'])
def command_help(message):
    my_bot.send_message(message.chat.id, "Бот Феечка Винкс умеет следующее:\n\n" + 
        "1. Если отправить ему песенку, то он выведет информацию о том, какой у нее жанр " +
        "(песни формата mp3 или wav)\n\n" +
        "2. (Самая главная функция): если ему отправить /meow, то он выдаст пикчу со случайной " +
        "кошкодевочкой с подписью, какой у нее мяу. Также есть три редкие девочки, " +
        "а какие, узнайте сами^^\n\n" +
        "3. Можно отправить голосовое сообщение, феечка также ответит, какой жанр у песни\n\n" +
        "Удачного пользования^^")


@my_bot.message_handler(commands=['meow'])
def command_meow(message):
    global last_meow
    print('@', end='')
    if message.chat.id not in last_meow:
        last_meow[message.chat.id] = []
    print(message.from_user.username, message.chat.id)

    was_send_meow = False
    for i in rare_anime_girl_meow:
        if (not was_send_meow) and (not randint(0, i[1])) and (last_meow[message.chat.id] != i):
            was_send_meow = True
            send_meow(i[0], i[2], message.chat.id)
            last_meow[message.chat.id] = i
    if not was_send_meow:
        girls = randint(0, len(common_anime_girl_meow) - 1)
        while last_meow[message.chat.id] == common_anime_girl_meow[girls]:
            girls = randint(0, len(common_anime_girl_meow) - 1)
        last_meow[message.chat.id] = common_anime_girl_meow[girls]
        send_meow(common_anime_girl_meow[girls][0], common_anime_girl_meow[girls][1], message.chat.id)

    
@my_bot.message_handler(content_types=['voice'])
def voice(message):
    my_bot.reply_to(message, "Я получил голосовуху")
    file_url = my_bot.get_file_url(message.voice.file_id)
    file_path = my_bot.get_file(message.voice.file_id).file_path
    format_file = file_path.split('.')[-1]
    f = save_file_on_pc(str(message.message_id) + '.' + format_file, file_url)
    genre = tensor_to_genre(spectograms_to_ganre(f))
    my_bot.reply_to(message, number_to_genre[genre][1])


@my_bot.message_handler(content_types=['photo'])
def photo(message):
    global need_spam
    my_bot.reply_to(message, 'Вы прислали картинку')
    if message.chat.id == main_chat_id and need_spam:
        send_spam(message.photo[0].file_id, message.caption)


@my_bot.message_handler(content_types=['text'])
def hype(message):
    my_bot.send_message(message.chat.id, "Вы прислали текст")


@my_bot.message_handler(content_types=['audio'])
def save_audio(message):
    my_bot.reply_to(message, "Вы прислали музыку")

    file_path = my_bot.get_file(message.audio.file_id).file_path
    format_audio = file_path.split('.')[-1]
    file_url = my_bot.get_file_url(message.audio.file_id)

    f = save_file_on_pc(str(message.message_id) + '.' + format_audio, file_url)
    my_bot.send_message(message.chat.id, "Музыка " + f.split('/')[1] + " успешно скачана")
    genre = tensor_to_genre(spectograms_to_ganre(f))
    my_bot.reply_to(message, number_to_genre[genre][1])


@my_bot.message_handler(content_types=['document'])
def save_document(message):
    my_bot.reply_to(message, "Вы прислали документ")
    file_url = my_bot.get_file_url(message.document.file_id)
    file_path = my_bot.get_file(message.document.file_id).file_path
    format_file = file_path.split('.')[-1]
    f = save_file_on_pc(str(message.message_id) + '.' + format_file, file_url)
    if format_file == 'wav':
        genre = tensor_to_genre(spectograms_to_ganre(f))
        my_bot.reply_to(message, number_to_genre[genre][1])
    my_bot.send_message(message.chat.id, "Документ " + f.split('/')[1] + " успешно скачан")


my_bot.polling()
