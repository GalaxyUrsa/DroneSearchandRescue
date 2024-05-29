from pynput import keyboard  
from multiprocessing import Process  
import time  

def DestoryThread(myThread):
    def on_press(key):
        try:
            if key.char=='q':
                # myThread.terminate()
                print("is destoryed")
        except:
            # print("not a char!")
            pass
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def listen_keyboard(valueset=None):  
    def on_press(key):
        try:
            if key.char=='y' and valueset[0]>0:
                valueset[0]-=1
                print(valueset)
            if key.char=='u' and valueset[0]<100:
                valueset[0]+=1
                print(valueset)
            if key.char=='h' and valueset[1]>0:
                valueset[1]-=1
                print(valueset)
            if key.char=='j' and valueset[1]<100:
                valueset[1]+=1
                print(valueset)
            if key.char=='n' and valueset[2]>10:
                valueset[2]-=1
                print(valueset)
            if key.char=='m' and valueset[2]<100:
                valueset[2]+=1
            
            print(valueset)
        except:
            print(f"Special key pressed: {key}")  
        # try:  
        #     print(f"Key pressed: {key.char}")  
        # except AttributeError:  
        #     print(f"Special key pressed: {key}")  
              
    # def on_release(key):  
    #     print(f"{key} release")  
    #     if key == keyboard.Key.esc:  
    #         # Stop listener  
    #         return False  
  
    # Collect events until released  
    # with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:  
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()  
  
def main():  
    # Create a subprocess to listen to the keyboard  
    p = Process(target=listen_keyboard)  
    p.start()  
      
    # Main process can continue doing other tasks  
    while True:  
        print("Main process is running...")  
        time.sleep(1)  