import requests
import json
import sys
import threading
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import numpy as np
from ultralytics import YOLO
import torch
import math
import cv2
import pyautogui
import chess

import speech_recognition as sr
import re
import pyttsx3


model = YOLO("yolov5nu.pt")

# Load the screenshot image
screenshot = pyautogui.screenshot()

# Predict the location of the chessboard in the screenshot
results = model(screenshot)


for x in results:
    x1, y1, x2, y2 = x.boxes.xyxy[0].numpy()




# Calculate the top-right and bottom-left corner coordinates
x3, y3 = x2, y1
x4, y4 = x1, y2

# Print all corner coordinates
top_left = [x1,y1] 

top_right =  [x3 ,y3]

bottom_right = [x2, y2]
bottom_left = [x4, y4]

BOARD_TOP_COORD = y1
BOARD_LEFT_COORD = x1


board_height = math.sqrt((bottom_left[0] - top_left[0]) ** 2 + (bottom_left[1] - top_left[1]) ** 2)

# Calculate the average dimension to get the board size
BOARD_SIZE = board_height
CELL_SIZE = int(BOARD_SIZE / 8)



print(f"Board size: {BOARD_SIZE}, Cell size: {CELL_SIZE}, Top coordinate: {BOARD_TOP_COORD}, Left coordinate: {BOARD_LEFT_COORD}")



model = load_model("keras_Model.h5", compile=False)
class_names = [
    "black_pawn",
    "black_king",
    "black_bishop",
    "black_queen",
    "black_rook",
    "black_knight",
    "white_pawn",
    "white_rook",
    "white_bishop",
    "white_knight",
    "white_king",
    "white_queen",
    "chessboard",
    "empty_squares"
]



def create_board():
    # Create an empty board
    board = np.zeros((8, 8), dtype=str)

    # Iterate over each cell in the board
    for row in range(8):
        for col in range(8):
            # Calculate the cell coordinates
            cell_top_coord = BOARD_TOP_COORD + row * CELL_SIZE
            cell_left_coord = BOARD_LEFT_COORD + col * CELL_SIZE

            # Crop the cell image
            screenshot = pyautogui.screenshot()
            cell_image = screenshot.crop(
                (cell_left_coord, cell_top_coord, cell_left_coord + CELL_SIZE, cell_top_coord + CELL_SIZE)
            )

            # Preprocess the cell image
            cell_image_grayscale = cell_image.convert("L")

            # Resize the cell image to match the expected input shape
            cell_image_resized = cell_image_grayscale.resize((224, 224), resample=Image.BILINEAR)

            # Convert the cell image to RGB
            cell_image_rgb = cell_image_resized.convert("RGB")

            cell_image_array = np.asarray(cell_image_rgb)
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = cell_image_array / 255.0

            # Predict the piece on the cell using the trained model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            percentage = round(prediction[0][index] * 100, 2)#added later
            piece = class_name.split("_")[1]  # get the piece name from the label
            color = class_name.split("_")[0]  # get the color name from the label
            piece_code = 0  # initialize to 0 (empty cell)
            if piece == "pawn":
                piece_code = 'P' if color == "white" else 'p'
            elif piece == "rook":
                piece_code = 'R' if color == "white" else 'r'
            elif piece == "knight":
                piece_code = 'N' if color == "white" else 'n'
            elif piece == "bishop":
                piece_code = 'B' if color == "white" else 'b'
            elif piece == "queen":
                piece_code = 'Q' if color == "white" else 'q'
            elif piece == "king":
                piece_code = 'K' if color == "white" else 'k'
            elif class_name == "empty_squares":
                piece_code = '.'
      

            # Set the piece code on the board
            board[row][col] = piece_code

    print(board)
    return board


def recognize_notation(r, engine, last_move):
    while True:
        with sr.Microphone() as source:
            print("Please say a valid chess notation")
            engine.say("Please say a valid chess notation")
            engine.runAndWait()

            r.adjust_for_ambient_noise(source)

            try:
                audio = r.listen(source, timeout=600, phrase_time_limit=3)
                text = r.recognize_google(audio)
                count=0
                notation=text
                if notation==83 or notation=="a tree":
                    notation=["A3"]
                elif notation=="even":
                    notation=["E1"]
                elif notation=="8":
                    if count%2==0:
                        notation=["A8"]
                        count+=1
                    else:
                        notation=["E8"]
                        count+=1
                elif notation=="before":
                    notation=["B4"]
                elif notation=="20 X":
                    notation=["B6"]
                elif notation=="beard":
                    notation=["B8"]
                elif notation=="c-cex":
                    notation=["C6"]
                elif notation=="Deewan":
                    notation=["D1"]
                elif notation=="de do":
                    notation=["D2"]
                elif notation=="D sex":
                    notation=["D6"]
                elif notation=="date":
                    notation = ["D8"]
                elif "MI" in notation:
                    match = re.search(r"MI\s+(\d+)", notation)
                    number = match.group(1)
                    newtext = "E"+str(number)
                    notation = [newtext]
                elif notation=="YouTube":
                    notation="E2"
                elif notation=="sex" or notation=="physics":
                    notation=["E6"]
                elif notation=="Jivan":
                    notation=["G1"]
                elif notation=="Jeetu" or notation=="Neetu":
                    notation=["G2"]
                elif notation=="zefo":
                    notation=["G4"]
                elif notation=="ji sex" or notation=="acchi sex":
                    notation=["G6"]
                else:
                    notation = re.findall(r'(?<!\w)(?:[a-hA-H][1-8]|last)', text, re.IGNORECASE)



                if notation:
                    if notation[0] != "last":
                        print(f"Recognized notation: {notation[0].upper()}")
                        engine.say(f"You said {notation[0]}, is that correct?")
                        engine.runAndWait()
                        confirm = recognize_confirmation(r, engine)
                    else:
                        confirm = False

                    if confirm:
                        return notation[0].lower()
                    else:
                        engine.say(f"Last Move is {last_move}")
                        engine.runAndWait()
                else:
                    print(f"Invalid notation: {text}")
                    engine.say("Sorry, I did not recognize a valid notation")
                    engine.runAndWait()

            except sr.WaitTimeoutError:
                print("Timeout occurred while waiting for phrase. Please speak again.")
                engine.say("Timeout occurred while waiting for phrase. Please speak again.")
                engine.runAndWait()

            except sr.UnknownValueError:
                print("Could not understand audio")
                engine.say("Sorry, I did not understand what you said")
                engine.runAndWait()

            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                engine.say("Sorry, I could not understand what you said")
                engine.runAndWait()




def recognize_confirmation(r, engine):
    while True:
        with sr.Microphone() as source:
            print("Please say 'yes' or 'no'")
            engine.say("Please say 'yes' or 'no'")
            engine.runAndWait()

            r.adjust_for_ambient_noise(source)

            try:
                audio = r.listen(source, timeout=15, phrase_time_limit=3)
                text = r.recognize_google(audio)
                if text.lower() in ['yes', 'yeah', 'correct', 'right']:
                    print("Confirmed")
                    engine.say("Confirmed")
                    engine.runAndWait()
                    return True
                elif text.lower() in ['no', 'nope', 'incorrect', 'wrong']:
                    print("Not confirmed")
                    engine.say("Not confirmed")
                    engine.runAndWait()
                    return False
                else:
                    print("Invalid confirmation")
                    engine.say("Sorry, I did not recognize a valid confirmation")
                    engine.runAndWait()
                    return False

            except sr.WaitTimeoutError:
                print("Could not understand audio")
                engine.say("Sorry, I did not understand what you said")
                engine.runAndWait()

            except sr.UnknownValueError:
                print("Could not understand audio")
                engine.say("Sorry, I did not understand what you said")
                engine.runAndWait()

            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                engine.say("Sorry, I could not understand what you said")
                engine.runAndWait()

        

def go(last_move, count):
    r = sr.Recognizer()
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) # setting the speaking rate to 150 words per minute
    notation = None

    if count:
        engine.say(f"Last Move is {last_move}")
        engine.runAndWait()

    while notation is None:
        notation = recognize_notation(r, engine,last_move)

    return notation









def move(last_move, color):

    # Ask for user input to specify the move
    start_position = go(last_move, count=1)
    end_position = go(last_move,count=0)

    # Calculate the start and end coordinates based on user input
    if color == "white":
        start_col = ord(start_position[0]) - ord('a')
        start_row = 8 - int(start_position[1])
        end_col = ord(end_position[0]) - ord('a')
        end_row = 8 - int(end_position[1])
    else:  
        start_col = ord('h') - ord(start_position[0])
        start_row = int(start_position[1]) - 1
        end_col = ord('h') - ord(end_position[0])
        end_row = int(end_position[1]) - 1

    # Calculate the start and end pixel coordinates
    start_x = BOARD_LEFT_COORD + start_col * CELL_SIZE + CELL_SIZE // 2
    start_y = BOARD_TOP_COORD + start_row * CELL_SIZE + CELL_SIZE // 2
    end_x = BOARD_LEFT_COORD + end_col * CELL_SIZE + CELL_SIZE // 2
    end_y = BOARD_TOP_COORD + end_row * CELL_SIZE + CELL_SIZE // 2

    # Simulate mouse movement and click to perform the move
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    pyautogui.moveTo(end_x, end_y)
    pyautogui.mouseUp()








api_token = "lip_yzGgMK9RQgJTo2aMUlHH"
username = "Happy02001"

def get_current_ongoing_game(username):
    try:
        headers = {
            "Accept": "application/json"
        }
        response = requests.get(f"https://lichess.org/api/user/{username}/current-game", headers=headers)
        if response.status_code == 200:
            game_data = response.json()
            return game_data
        else:
            print(f"Failed to retrieve the current ongoing game for {username}.")
            return None
    except requests.RequestException as e:
        print(f"An error occurred while retrieving the current ongoing game: {str(e)}")
        return None


current_ongoing_game = get_current_ongoing_game(username)


game_id=""

if current_ongoing_game:
    game_id = current_ongoing_game['id']
    


def get_ongoing_games(api_token):
    try:
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json"
        }
        response = requests.get("https://lichess.org/api/account/playing", headers=headers)
        if response.status_code == 200:
            games_data = response.json()
            return games_data.get("nowPlaying", [])
        else:
            print("Failed to retrieve ongoing games.")
            return []
    except requests.RequestException as e:
        print(f"An error occurred while retrieving ongoing games: {str(e)}")
        return []






def determine_turn(game):
    if game.get("isMyTurn"):
        return "Your turn"
    else:
        return "Opponent's turn"
    



def stream_game_moves(api_token, game_id, initial_gamefull_state_printed = False):
    url = f"https://lichess.org/api/board/game/stream/{game_id}"
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code == 200:
        last_move = None

        for line in response.iter_lines():
            if line:
                event_data = line.decode('utf-8')
                event_json = json.loads(event_data)

                if not initial_gamefull_state_printed:
                    event_json = event_json.get('state', None)
                    event_type = event_json.get('type', None)
                    initial_gamefull_state_printed = True
                    ongoing_games = get_ongoing_games(api_token)
                    if ongoing_games:
                        for game in ongoing_games:
                            turn = determine_turn(game)
                            if game['gameId'] in game_id:
                                moves = event_json.get('moves', '')
                                move_list = moves.split()
                                if turn[:1].lower() == "y" and game.get('color', None)!="black" and len(move_list)==0:
                                    print(turn)
                                    r = sr.Recognizer()
                                    engine = pyttsx3.init()
                                    engine.setProperty('rate', 150)
                                    engine.say(turn)
                                    engine.runAndWait()
                                    move(last_move, game['color'])


                if event_type == 'gameState':
                    if 'moves' in event_json:
                        moves = event_json['moves']
                        move_list = moves.split()
                        if move_list:
                            # Check if the last move is played by the opponent
                            if last_move is None or move_list[-1] != last_move:
                                last_move = move_list[-1]
                                
                                ongoing_games = get_ongoing_games(api_token)
                                if ongoing_games:
                                    for game in ongoing_games:
                                        turn = determine_turn(game)
                                        if game['gameId'] in game_id:
                                            if turn[:1].lower() == "y":
                                                board = create_board()
                                                print(f"Opponent played: {last_move}")
                                                move(last_move, game['color'])


                                else:
                                    print("No ongoing games found.")


        
        r = sr.Recognizer()
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) 
        

        if event_json.get('status') == 'mate':
            winner = event_json.get('winner', 'none')
            if winner == 'white':
                    
                print("White player won the game.")
                engine.say("White player won the game.")
                engine.runAndWait()

            elif winner == 'black':
                print("Black player won the game.")
                engine.say("Black player won the game.")
                engine.runAndWait()


        elif event_json.get('status') == 'resign':
            winner = event_json.get('winner')
            print("came to it")
            if winner == 'black':
                print("White player resigned.")
                engine.say("White player resigned.")
                engine.runAndWait()


            elif winner == 'white':
                print("Black player resigned.")
                engine.say("Black player resigned.")
                engine.runAndWait()



        else:
            print("Game ended in a draw")
            engine.say("Game ended in a draw")
            engine.runAndWait()
            
            

    else:
        print(f"Error: {response.status_code} - {response.text}")


stream_game_moves(api_token, game_id)






