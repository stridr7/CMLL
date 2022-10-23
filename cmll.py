import cv2, re, os, json, sys, win32api, nltk, pickle, time, datetime, random 
import speech_recognition, face_recognition, urllib.request, feedparser, pytesseract
import numpy as np, pyttsx3 as vks, webbrowser as wolfet, keyboard as key

from urllib.parse  import urlencode
from PIL           import Image, ImageDraw
from selenium      import webdriver
from abc           import ABCMeta, abstractmethod
from nltk.stem     import WordNetLemmatizer

from tensorflow.python.keras.layers    import Dense
from tensorflow.keras.models           import Sequential
from tensorflow.keras.layers           import Dense, Dropout
from tensorflow.keras.optimizers       import SGD
from tensorflow.keras.models           import load_model
from webdriver_manager.chrome          import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
pytesseract.pytesseract.tesseract_cmd = r'C:\...\tesseract.exe'
recognizer = speech_recognition.Recognizer()
sound = vks.init()
sound.setProperty('rate',186)
agenda    = ['']
name_list = ['']
words     = pickle.load(open('words.pkl', 'rb'))
classes   = pickle.load(open('classes.pkl', 'rb'))
model     = load_model('model.h5')


pass
class IAssistant(metaclass=ABCMeta):

    @abstractmethod
    def train_model(self):
        """ Implemented in child class """

    @abstractmethod
    def request_tag(self, message):
        """ Implemented in child class """

    @abstractmethod
    def get_tag_by_id(self, id):
        """ Implemented in child class """

    @abstractmethod
    def request_method(self, message):
        """ Implemented in child class """

    @abstractmethod
    def request(self, message):
        """ Implemented in child class """

class GenericAssistant(IAssistant):

    def __init__(self, intents, intent_methods={}, model_name="assistant_model", *, json_encoding='utf-8-sig'):
        self.intents = intents
        self.intent_methods = intent_methods
        self.model_name = model_name
        self.json_encoding = json_encoding

        if intents.endswith(".json"):
            self.load_json_intents(intents)

        self.lemmatizer = WordNetLemmatizer()

    def load_json_intents(self, intents):
        with open(intents, encoding=self.json_encoding) as f:
            self.intents = json.load(f)

    def train_model(self):
        self.words = []
        self.classes = []
        documents = []
        ignore_letters = ['!', '?', ',', '.']

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))

        self.classes = sorted(list(set(self.classes)))



        training = []
        output_empty = [0] * len(self.classes)

        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    def save_model(self, model_name=None):
        if model_name is None:
            self.model.save(f"{self.model_name}.h5", self.hist)
            with open(f'{self.model_name}_words.pkl', 'wb') as f:
                pickle.dump(self.words, f)
            with open(f'{self.model_name}_classes.pkl', 'wb') as f:
                pickle.dump(self.classes, f)
        else:
            self.model.save(f"{model_name}.h5", self.hist)
            with open(f'{model_name}_words.pkl', 'wb') as f:
                pickle.dump(self.words, f)
            with open(f'{model_name}_classes.pkl', 'wb') as f:
                pickle.dump(self.classes, f)

    def load_model(self, model_name=None):
        if model_name is None:
            self.words = pickle.load(open(f'{self.model_name}_words.pkl', 'rb'))
            with open(f'{self.model_name}_words.pkl', 'rb') as f:
                self.words = pickle.load(f)
            with open(f'{self.model_name}_classes.pkl', 'rb') as f:
                self.classes = pickle.load(f)
            self.model = load_model(f'{self.model_name}.h5')
        else:
            with open(f'{model_name}_words.pkl', 'rb') as f:
                self.words = pickle.load(f)
            with open(f'{model_name}_classes.pkl', 'rb') as f:
                self.classes = pickle.load(f)
            self.model = load_model(f'{model_name}.h5')

    def _clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, sentence, words):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def _predict_class(self, sentence):
        p = self._bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def _get_response(self, ints, intents_json):
        try:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag']  == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            result = "I don't understand!"
        return result

    def request_tag(self, message):
        pass

    def get_tag_by_id(self, id):
        pass

    def request_method(self, message):
        pass

    def request(self, message):
        ints = self._predict_class(message)

        if ints[0]['intent'] in self.intent_methods.keys():
            self.intent_methods[ints[0]['intent']]()
        else:
            sound.say(self._get_response(ints, self.intents))
  
def identify():
    global recognizer
    sound.say('')
    sound.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration = 3)
                audio = recognizer.listen(mic)
                yourname = recognizer.recognize_google(audio, language = 'pt-BR')
                yourname = yourname.lower()
                camera = cv2.VideoCapture(0)
                i = 0
                im_path = "C:\\Users\\Public\\"
                while i < 1:
                    return_value, image = camera.read()
                    cv2.imwrite(os.path.join(im_path, yourname +'.jpg'), image)
                    i += 1
                del(camera)
                name_list.append(yourname)
                sound.say(f'..., {yourname}')
                sound.runAndWait()
                done = True
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            sound.say('')
            sound.runAndWait()
            
def nameint():
    sound.say('')
    camera = cv2.VideoCapture(0)
    i = 0
    im_path = ""
    while i < 1:
        return_value, image = camera.read()
        cv2.imwrite(os.path.join(im_path,'temp' + '.jpg'), image)
        i += 1
    del(camera)
    
    Ximage = face_recognition.load_image_file(" ")
    Ximage_face_encoding = face_recognition.face_encodings(Ximage)[0]

    test_image = face_recognition.load_image_file(" ")
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    pil_image = Image.fromarray(test_image)
    draw = ImageDraw.Draw(pil_image)

    for(top, right, bottom, left), face_encoding in zip (face_locations, face_encodings):
        matches = face_recognition.compare_faces(name_list, face_encoding)
        name = "Unknown Person"

        if True in matches:
            first_match_index = matches.index(True)
            name = name_list[first_match_index]
            sound.say(" " + name)
            sound.runAndWait()
        else:
            sound.say(' ')
            sound.runAndWait()
    

    os.remove(" ")
    camB = cv2.VideoCapture(0)
    camB.release        

def add_agenda():
    global recognizer
    sound.say(' ')
    sound.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration = 3)
                audio = recognizer.listen(mic)
                item = recognizer.recognize_google(audio, language = 'pt-BR')
                item = item.lower()
                agenda.append(item)
                done = True
                sound.say(f' ')
                sound.runAndWait()
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            sound.say(' ')
            sound.runAndWait()

def addcontact():
    global recognizer
    sound.say(' ')
    sound.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration = 3)
                audio = recognizer.listen(mic)
                ctname = recognizer.recognize_google(audio, language = 'pt-BR')
                ctname = ctname.lower()
                sound.say(' ')
                sound.runAndWait()
                audio = recognizer.listen(mic)
                codechecker = recognizer.recognize_google(audio, language = 'pt-BR')
                codechecker = codechecker.lower()
                if codechecker == ' ' or codechecker == ' ' or codechecker == ' ' or codechecker == ' ' or codechecker == ' ':
                    codeno = ' '
                else:
                    codeno = ''
                    sound.say(' ')
                    sound.runAndWait()
                sound.say(' ')
                sound.runAndWait()
                audio = recognizer.listen(mic)
                ctno = recognizer.recognize_google(audio, language = 'pt-BR')
                ctno = ctno.lower()
                ctno = ctno.replace(' ','')
                ctno = ctno.replace('zero','0')
                ctno = ctno.replace('um','1')
                ctno = ctno.replace('dois','2')
                ctno = ctno.replace('três','3')
                ctno = ctno.replace('tres','3')
                ctno = ctno.replace('quatro','4')
                ctno = ctno.replace('cinco','5')
                ctno = ctno.replace('seis','6')
                ctno = ctno.replace('sete','7')
                ctno = ctno.replace('oito','8')
                ctno = ctno.replace('nove','9')
                ctno = ctno.replace('dez','10')
                ctno = ctno.replace('onze','11')
                ctno = ctno.replace('doze','12')
                ctno = ctno.replace('treze','13')
                ctno = ctno.replace('catorze','14')
                ctno = ctno.replace('quinze','15')
                ctno = ctno.replace('dezesseis','16')
                ctno = ctno.replace('dezessete','17')
                ctno = ctno.replace('dezoito','18')
                ctno = ctno.replace('dezenove','19')
                ctno = ctno.replace('vinte','20')
                ctno = ctno.replace('vinte e ','2')
                ctno = ctno.replace('trinta','30')
                ctno = ctno.replace('trinta e ','3')
                ctno = ctno.replace('quarenta','40')
                ctno = ctno.replace('quarenta e ','4')
                ctno = ctno.replace('cinquenta','50')
                ctno = ctno.replace('cinquenta e ','5')
                ctno = ctno.replace('sessenta','60')
                ctno = ctno.replace('sessenta e ','6')
                ctno = ctno.replace('setenta','70')
                ctno = ctno.replace('setenta e ','7')
                ctno = ctno.replace('oitenta','80')
                ctno = ctno.replace('oitenta e ','8')
                ctno = ctno.replace('noventa','90')
                ctno = ctno.replace('noventa e ','9')
                with open(ctname + '.txt', 'w') as contactfile:
                    contactfile.write(codeno + ctno)
                    contactfile.close()
                sound.say(f' ')
                sound.runAndWait()
                break
                    
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            sound.say(' ')
            sound.runAndWait()
            
def whatsto():
    global recognizer
    sound.say(' ')
    sound.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration = 3)
                audio = recognizer.listen(mic)
                wcontact = recognizer.recognize_google(audio, language = 'pt-BR')
                wcontact = wcontact.lower()
                try:
                    with open(wcontact + '.txt', 'r') as contfile:
                        contnumber = contfile.read()
                        done = True
                except:
                    sound.say(' ?')
                    sound.runAndWait()
                    break
                sound.say(' ')
                sound.runAndWait()                
                recognizer.adjust_for_ambient_noise(mic, duration = 5)
                audio = recognizer.listen(mic)
                wmescontent = recognizer.recognize_google(audio, language = 'pt-BR')
                wmescontent = wmescontent.lower()
                chrome_options = Options()
                chrome_options.add_argument(r"user-data-dir=" + "c://users//public//user_data")
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
                driver.set_window_position(-10000,0)
                driver.get(f'https://web.whatsapp.com/send?phone={contnumber}&text={wmescontent}')
                time.sleep(5)
                key.press_and_release('enter')
                key.press_and_release('alt,space')
                key.press_and_release('n')
                done = True                
                
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            sound.say(' ')
            sound.runAndWait()

def playmus():
    global recognizer
    sound.say(' ')
    sound.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration = 3)
                audio = recognizer.listen(mic)
                songname = recognizer.recognize_google(audio, language = 'pt-BR')
                songname = songname.lower()
                getmedia = songname
                getmedia = getmedia.replace(" ","+")
                html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + getmedia)
                video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
                wolfet.open("https://www.youtube.com/watch?v=" + video_ids[0])
                break
                
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            sound.say(' ')
            sound.runAndWait()
     
def findnet():
    global recognizer
    sound.say('t ')
    sound.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration = 3)
                audio = recognizer.listen(mic)
                therm = recognizer.recognize_google(audio, language = 'pt-BR')
                therm = therm.lower()
                therms = therm
                therms = therms.replace(" ","+")    
                wolfet.open('https://www.google.com/search?q=' + therms)
                done = True
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            sound.say(' ')
            sound.runAndWait()    
        
def translator():
    global recognizer
    sound.say(' ')
    sound.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration = 3)
                audio = recognizer.listen(mic)
                totransl = recognizer.recognize_google(audio, language = 'pt-BR')
                totransl = totransl.lower()
                if totransl == ' ' or totransl == ' ':
                    sound.say(' ')
                    sound.runAndWait()
                    recognizer.adjust_for_ambient_noise(mic, duration = 3)
                    audio = recognizer.listen(mic)
                    phrtotr = recognizer.recognize_google(audio)
                    phrtotr = phrtotr.lower()
                    phrtotr = phrtotr.replace(" ", "%20")
                    wolfet.open(f'https://translate.google.com/?sl=auto&tl=&text={phrtotr}%0A&op=translate')
                    sound.say('aqui está')
                    sound.runAndWait()
                    done = True
                elif totransl == ' ' or totransl == ' ' or totransl == ' ' or totransl == ' ' or totransl == ' ':
                    sound.say(' ')
                    sound.runAndWait()
                    camera = cv2.VideoCapture(0)
                    i = 0
                    im_path = "C:\\Users\\Public\\image"
                    while i < 1:
                        return_value, image = camera.read()
                        cv2.imwrite(os.path.join(im_path,'ift' + '.jpg'), image)
                        i += 1
                    del(camera)
                    sound.say(' ')
                    sound.runAndWait()
                    context = pytesseract.image_to_string('c:\\users\\public\\ift.jpg')
                    context = context.replace(" ", "%20")
                    wolfet.open(f'https://translate.google.com/?sl=auto&tl=pt&text={context}%0A&op=translate')
                    sound.say(' ')
                    sound.runAndWait()
                    os.remove('c:\\users\\public\\ift.jpg')
                    done = True                    

                else:
                    sound.say(' ')
                    sound.runAndWait()
                    break
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            sound.say(' ')
            sound.runAndWait()
            break

def readnews():
    sound.say(' ')
    sound.runAndWait()
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration = 3)
                audio = recognizer.listen(mic)
                news_source = recognizer.recognize_google(audio, language = 'pt-BR')
                news_source = news_source.lower()
                if news_source == ' ' or news_source == ' ':
                    target = xywz
                elif news_source == ' ':
                    target = stuv
                
                path = ('//RSS//')

                with open (path + target + '.txt', 'r') as url:
                    seekcontent = url.read()
                    feed = feedparser.parse(seekcontent)
                    feedcontent = feed.entries[1]
                    sound.say(feedcontent)
                    sound.runAndWait()

        except:
            sound.say(' ')
            sound.runAndWait()
            break

    
def stpgstr():
    hande = 1
    sound.say(' ')
    sound.runAndWait()
    os.system(' ')
    
def gestr():
    import gestr
    sound.say(' ')
    sound.runAndWait()
    if hande == 1:
        return None
    else:
        os.system(' e')
                         
def getdate():
    currentday = datetime.datetime.now()
    curday = currentday.day
    curmon = currentday.month 
    curyar = currentday.year
    sound.say(f'hoje é dia {curday} de {curmon} de {curyar}')
    sound.runAndWait()

def gethour():
    currenthour = datetime.datetime.now()
    curhour = currenthour.hour
    curmins = currenthour.minute
    if curhour == '01' or curhour == '1':
        hourdef = 'hora'
    else:
        hourdef = 'horas'
    if curmins == '01':
        mindef = 'minuto'
    else:
        mindef = 'minutos'

    sound.say(f'{curhour} {hourdef} e {curmins} {mindef}')
    sound.runAndWait()
        
def read_agenda():
    sound.say(" ")
    for item in agenda:
        sound.say(item)
        sound.runAndWait()

def close():
    sound.say(' ')
    sound.runAndWait()
    sys.exit(0)

def totalshutdown():
    sound.say(' ')
    sound.runAndWait()
    os.system('shutdown -s -t 5')

mappings = {
 
    "addcontact"  : addcontact,
    "add_agenda"  : add_agenda,
    "read_agenda" : read_agenda,
    "playmusic"   : playmus,
    "finditem"    : findnet,
    "translator"  : translator,
    "getdate"     : getdate,
    "gethour"     : gethour,
    "messageto"   : whatsto,
    "turnoffsys"  : totalshutdown,
    "finish"      : close,
    "news"        : readnews,
    "gesture"     : gestr,
    "stpgesture"  : stpgstr,
    "identify"    : getname,
    "nameint"     : nameint
    }

assistant = GenericAssistant('intents.json', intent_methods = mappings)
assistant.train_model()    

while True:
    try:
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration = 7)
            audio = recognizer.listen(mic)
            message = recognizer.recognize_google(audio, language = 'pt-BR')
            message = message.lower()
            assistant.request(message)
    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()
        sound.say('')
        sound.runAndWait()
