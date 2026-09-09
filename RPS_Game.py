import cv2
import mediapipe as mp
import random
import time
from collections import deque
import statistics as st


# ---------------------------------------------------------------------------
# Game logic helpers
# ---------------------------------------------------------------------------

def calculate_winner(cpu_choice, player_choice):
    """Determines the winner of each round given the CPU's and player's moves."""

    if player_choice == "Invalid":
        return "Invalid!"

    if player_choice == cpu_choice:
        return "Tie!"

    elif player_choice == "Rock" and cpu_choice == "Scissors":
        return "You win!"

    elif player_choice == "Rock" and cpu_choice == "Paper":
        return "CPU wins!"

    elif player_choice == "Scissors" and cpu_choice == "Rock":
        return "CPU wins!"

    elif player_choice == "Scissors" and cpu_choice == "Paper":
        return "You win!"

    elif player_choice == "Paper" and cpu_choice == "Rock":
        return "You win!"

    elif player_choice == "Paper" and cpu_choice == "Scissors":
        return "CPU wins!"

    # Defensive fallback – should never be reached in normal play
    return "Invalid!"


def compute_fingers(hand_landmarks, count):
    """
    Counts the number of extended fingers for a single hand.

    hand_landmarks: list of [id, xPos, yPos, label] for one hand only.
    Coordinates are used to determine whether a finger is extended by
    comparing the fingertip position to the middle-joint position.
    For the thumb the x-axis is used (direction depends on handedness).
    """

    # Index Finger (tip=8, pip=6)
    if hand_landmarks[8][2] < hand_landmarks[6][2]:
        count += 1

    # Middle Finger (tip=12, pip=10)
    if hand_landmarks[12][2] < hand_landmarks[10][2]:
        count += 1

    # Ring Finger (tip=16, pip=14)
    if hand_landmarks[16][2] < hand_landmarks[14][2]:
        count += 1

    # Pinky Finger (tip=20, pip=18)
    if hand_landmarks[20][2] < hand_landmarks[18][2]:
        count += 1

    # Thumb – uses x-axis; direction flips based on handedness
    if hand_landmarks[4][3] == "Left" and hand_landmarks[4][1] > hand_landmarks[3][1]:
        count += 1
    elif hand_landmarks[4][3] == "Right" and hand_landmarks[4][1] < hand_landmarks[3][1]:
        count += 1

    return count


# ---------------------------------------------------------------------------
# Drawing helpers (dynamic, resolution-independent positioning)
# ---------------------------------------------------------------------------

def put_text(image, text, rel_x, rel_y, scale=1.8, color=(255, 255, 255), thickness=4):
    """
    Draw text at a position expressed as fractions of the image dimensions.
    rel_x and rel_y are in the range [0, 1].
    """
    h, w = image.shape[:2]
    x = int(rel_x * w)
    y = int(rel_y * h)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, text, (x, y), font, scale, color, thickness)


def draw_overlay(image, player_choice, cpu_choice, winner, winner_colour,
                 player_score, cpu_score, countdown_text):
    """Render all HUD elements onto the frame."""
    h, w = image.shape[:2]

    # ------ Labels ------
    put_text(image, "You",        0.06, 0.12, color=(255, 80, 80),  thickness=5)
    put_text(image, "CPU",        0.78, 0.12, color=(80, 80, 255),  thickness=5)

    # ------ Scores ------
    put_text(image, str(player_score), 0.10, 0.30, color=(255, 80, 80),  thickness=5)
    put_text(image, str(cpu_score),    0.82, 0.30, color=(80, 80, 255),  thickness=5)

    # ------ Moves ------
    put_text(image, player_choice, 0.04, 0.58, color=(255, 80, 80),  thickness=5)
    put_text(image, cpu_choice,    0.72, 0.58, color=(80, 80, 255),  thickness=5)

    # ------ Countdown or winner ------
    if countdown_text:
        put_text(image, countdown_text, 0.43, 0.88, scale=2.5,
                 color=(0, 220, 255), thickness=6)
    else:
        put_text(image, winner,    0.38, 0.88, color=winner_colour, thickness=5)

    # ------ Instructions (small, bottom-left) ------
    cv2.putText(image, "ESC: Quit | R: Reset", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


# ---------------------------------------------------------------------------
# Game state helpers
# ---------------------------------------------------------------------------

def reset_game(cpu_score, player_score):
    """Return a fresh game state; optionally preserve scores if caller passes 0s."""
    return dict(
        cpu_choice="Nothing",
        player_choice="Nothing",
        winner="None",
        winner_colour=(0, 200, 0),
        cpu_score=cpu_score,
        player_score=player_score,
        hand_valid=False,
        countdown_start=None,   # wall-clock time when countdown began
        de=deque(["Nothing"] * 5, maxlen=5),
    )


COUNTDOWN_SECONDS = 3
# Maps finger count → move name (index = finger count 0-5)
DISPLAY_VALUES = ["Rock", "Invalid", "Scissors", "Invalid", "Invalid", "Paper"]


# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

webcam = cv2.VideoCapture(0)

state = reset_game(0, 0)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while webcam.isOpened():
        success, image = webcam.read()
        if not success:
            print("Camera isn't working")
            continue

        # Mirror the image so it feels natural
        image = cv2.flip(image, 1)

        # Hand detection (MediaPipe expects RGB)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ----------------------------------------------------------------
        # Hand landmark processing
        # ----------------------------------------------------------------
        count = 0
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True

            for hand_idx, hand in enumerate(results.multi_hand_landmarks):
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    image,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                label = results.multi_handedness[hand_idx].classification[0].label
                imgH, imgW, _ = image.shape

                # Build landmark list for THIS hand only (reset per hand)
                hand_landmarks = []
                for landmark_id, landmark in enumerate(hand.landmark):
                    xPos = int(landmark.x * imgW)
                    yPos = int(landmark.y * imgH)
                    hand_landmarks.append([landmark_id, xPos, yPos, label])

                # Accumulate finger count across all detected hands
                count = compute_fingers(hand_landmarks, count)

        # ----------------------------------------------------------------
        # Determine raw player choice from finger count
        # ----------------------------------------------------------------
        if hand_detected:
            if count <= 5:
                raw_choice = DISPLAY_VALUES[count]
            else:
                raw_choice = "Invalid"
        else:
            raw_choice = "Nothing"

        # Smooth choice using a mode over the last 5 frames
        state["de"].appendleft(raw_choice)
        try:
            state["player_choice"] = st.mode(state["de"])
        except st.StatisticsError:
            state["player_choice"] = raw_choice

        # ----------------------------------------------------------------
        # State machine: WAITING → COUNTDOWN → RESOLVING
        # ----------------------------------------------------------------
        now = time.time()
        countdown_text = ""

        if not hand_detected:
            # Hand left frame — reset round state so a new round can begin
            state["hand_valid"] = False
            state["countdown_start"] = None

        elif not state["hand_valid"]:
            # Hand is in frame and a round hasn't been resolved yet

            if state["player_choice"] == "Nothing":
                # Mode still catching up — don't start countdown yet
                state["countdown_start"] = None

            else:
                # Start countdown on first valid detection
                if state["countdown_start"] is None:
                    state["countdown_start"] = now

                elapsed = now - state["countdown_start"]
                remaining = COUNTDOWN_SECONDS - elapsed

                if remaining > 0:
                    # Show countdown
                    countdown_text = str(int(remaining) + 1)
                else:
                    # Countdown finished — resolve the round
                    state["hand_valid"] = True
                    state["cpu_choice"] = random.choice(["Rock", "Paper", "Scissors"])
                    state["winner"] = calculate_winner(
                        state["cpu_choice"], state["player_choice"])

                    if state["winner"] == "You win!":
                        state["player_score"] += 1
                        state["winner_colour"] = (255, 100, 100)
                    elif state["winner"] == "CPU wins!":
                        state["cpu_score"] += 1
                        state["winner_colour"] = (100, 100, 255)
                    else:
                        state["winner_colour"] = (0, 200, 0)

        # ----------------------------------------------------------------
        # Draw HUD
        # ----------------------------------------------------------------
        draw_overlay(
            image,
            player_choice=state["player_choice"],
            cpu_choice=state["cpu_choice"],
            winner=state["winner"],
            winner_colour=state["winner_colour"],
            player_score=state["player_score"],
            cpu_score=state["cpu_score"],
            countdown_text=countdown_text,
        )

        cv2.imshow("Rock, Paper, Scissors", image)

        # ----------------------------------------------------------------
        # Key handling
        # ----------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27:           # ESC – quit
            break
        elif key in (ord("r"), ord("R")):   # R – reset scores
            state = reset_game(0, 0)

webcam.release()
cv2.destroyAllWindows()
