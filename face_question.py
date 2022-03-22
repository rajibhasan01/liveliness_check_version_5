import random
import cv2
import time


question_bank = [
    "right",
    "left",
    "up"            
                 ];

# Generate global status 
def generate_status():
    global status;
    global counter_start;
    global count_for_call;
    global countdown;
    global question_index;
    global question;
    global final_result_for_all_qstn;
    global buffer_result_for_single_qstn;
    counter_start = time.time();
    status = None;
    count_for_call = 0;
    countdown = 0;
    question_index = random.randint(0,2);
    question = question_bank[question_index];
    final_result_for_all_qstn = [];
    buffer_result_for_single_qstn = [];
    
    final_result_for_all_qstn.clear();
    buffer_result_for_single_qstn.clear();

# Generating new question
def generate_qstn(img):
    global question_index;
    global countdown;
    global question;
    global counter_start;
    global status;
    global count_for_call;
        
    current_time = time.time();
    time_counter = generate_timer(current_time);
    
    if counter_start + 3.5 < current_time and status == None:
        
        # calculate previous question answer
        if status == None:
            # print("prev_qstn", question)
            calculate_result(question);
        
        new_question_index = random.randint(0,2);
        if question_index == new_question_index:
            if question_index == 0:
                new_question_index = question_index + 1;
            else:
                new_question_index = question_index - 1;
                
        counter_start = current_time;
        countdown = 0;
        question_index = new_question_index;
        question = question_bank[question_index];
        
        
    
    # check status
    if status != None:
        display_result(img);
       
    else:
        display_question(img, question, time_counter);
        return question;


#Edit image frame with question
def display_question(img, question, timer):
    
    if question == 'open':
        text = "Open Your Mouth";
    else:
        text = "Turn Your Face " + str(question);
    cv2.rectangle(img, (100, 10), (480, 50), (0, 255, 0), cv2.FILLED);
    cv2.putText(img,f'Timer: {timer}',(510,35),cv2.FONT_HERSHEY_COMPLEX,0.7,(0, 65, 255),1);
    cv2.putText(img,text,(110,40),cv2.FONT_HERSHEY_COMPLEX,1,(255, 0, 0),2);
    return img;

# Edit image frame with result
def display_result(img):
    global status;
    if status == "Real":
        bg = (106, 176, 76);
        txt_position = (260,40)
    else:
        bg = (0, 0, 205);
        txt_position = (260,40);
    cv2.rectangle(img, (170, 10), (440, 50), bg, cv2.FILLED);
    cv2.putText(img, status,txt_position,cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2);
    return img;
    

# Check match for every buffering question
def match_q_a(given_face_orientation):
    
    global buffer_result_for_single_qstn;
    global counter_start;
    buffer_result_for_single_qstn.append(given_face_orientation);
    

# Generating countdown for each question
def generate_timer(current_time):
    
    global countdown;
    global counter_start;
    
    if  counter_start + 1 < current_time:
        countdown += 1;
        return countdown;
    else:
        return countdown;


# Result Calculation
def calculate_result(question):
    
    global final_result_for_all_qstn;
    global buffer_result_for_single_qstn;
    global question_bank;
        
    if buffer_result_for_single_qstn.count(question):
        false_value_count = 0
        for qstn in question_bank:
            if qstn == question or qstn == 'font':
                continue;
            else:
                if buffer_result_for_single_qstn.count(qstn):
                    false_value_count += 1;
        if false_value_count:
            final_result_for_all_qstn.append(0);
            print("false")
        else:
            final_result_for_all_qstn.append(1);
            print("true");
            
    else:
        final_result_for_all_qstn.append(0);
        print("false");
    
    buffer_result_for_single_qstn.clear();
    
    if len(final_result_for_all_qstn) == 4:
        make_decision(final_result_for_all_qstn);
        final_result_for_all_qstn.clear();
        print(final_result_for_all_qstn);


# Make a decision for varified or not
def make_decision(final_result_for_all_qstn):
    
    global status;
    rignt_ans_count = final_result_for_all_qstn.count(1);
    
    if(rignt_ans_count >= 3):
        status = "Real"
        print("Passed")
    else:
        status = "Fake"
        print("failed")