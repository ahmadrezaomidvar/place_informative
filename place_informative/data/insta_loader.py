from instaloader import Instaloader, Profile, load_structure_from_file, save_structure_to_file, RateController
import time
import random
import datetime
import json
import os
from pathlib import Path
from sacred import Experiment

ex = Experiment('insta_download')



# class to customizing of RateConntroller for better handeling of 429 error
class CustomRateController(RateController):

    def sleep(self, secs: float):
        """Wait given number of seconds."""
        # Not static, to allow for the behavior of this method to depend on context-inherent properties, such as
        # whether we are logged in.
        # pylint:disable=no-self-use
        print('custom handling of 429 error, sleeping for 1 seconds')
        time.sleep(1)

class DownloadProfile:

    def __init__(self, _config):
        '''
        ** Instaloader keywords specified in config.yaml file**
        '''
        pass
        

    @ex.capture
    def _make_loader(self, profile_name, download_dir, download_videos=False, download_comments=False, download_geotags = False, save_metadata = False, 
                    post_metadata_txt_pattern='', login_users_path=None, max_connection_attempts=0, request_timeout=300):

        # selecting the user in case loging_user file specified
        if login_users_path:
            with open(login_users_path) as json_file:
                self.login_users = json.load(json_file)
                self.users = []
                for key in self.login_users:
                    self.users.append(key)
                user, password= self.change_user(self.users, self.login_users)
        
        # loader creation
        self.loader = Instaloader(
                                download_videos=download_videos,
                                download_comments=download_comments,
                                download_geotags=download_geotags,
                                save_metadata=save_metadata,
                                dirname_pattern = f'{download_dir}/{profile_name}',
                                post_metadata_txt_pattern = post_metadata_txt_pattern,
                                max_connection_attempts = max_connection_attempts,
                                request_timeout = request_timeout,
                                rate_controller=lambda ctx: CustomRateController(ctx)
                                )
        print('loader created sucessfully')                                                                                  
        if login_users_path:                       
            self._loader_login(user, password)

    # login function
    @ex.capture
    def _loader_login(self, user, password):
        self.loader.login(user, password)

    # profile making function
    @ex.capture
    def _make_profile(self, profile_name):

        self.profile =  Profile.from_username(self.loader.context, profile_name)
        print('profile created sucessfully')                                                                                  

    # main download function
    @ex.capture
    def download(self, profile_name, download_dir, night_sleep, morning_wake, night_sleep_time, 
                 max_download, min_time, down_sleep_min, down_sleep_max,
                 random_sleep_time_start, random_sleep_time_end, random_probability,
                 proxies_path, login_users_path=None):
        
        # list the proxies specified in proxies_path in config.yaml file
        with open(proxies_path) as x:
            proxies = x.read().split('\n')
            try:
                proxies.remove('')          # try to remove extra lines if exists
            except Exception:
                pass
        
        i = 0
        # making Instaloader and profile objects
        while True:
            try:
                i = self.change_proxy(proxies, i)
                self._make_loader(profile_name)
                self._make_profile(profile_name)
                break

            except Exception:
                print('either loader or profile could not be created. retrying...')
                pass
        
        response = False    # response to download. True if download is sucessfull and False if the file exists
        counter = 0         # counter to control sleeping and randomness behaviour
        retry = 0           # counter to try to exit if the code stuck in an specific post
        start = time.time()

        all_posts = self.profile.get_posts()
        path_to_resume = Path(f'{download_dir}/{profile_name}/resume_information.json')
        self.remove_resume(str(path_to_resume))      # will check if broken resume_information.json exist if the code is interrupted in previous run
        print(f"\nscraping '{profile_name}', totally {all_posts.count} posts detected...")


        while (all_posts.total_index < all_posts.count and retry < (all_posts.count * 10)):      # download will be finished if all posts are downloaded or code try to download all posts more than 10 times of all posts quantity.

            try:
                retry += 1

                # check if resume_information.json exists and load the structure from it.
                if path_to_resume.exists():
                    resume_info = load_structure_from_file(self.loader.context, str(path_to_resume))
                    all_posts = self.profile.get_posts()
                    all_posts.thaw(resume_info)
                    print(f'resuming download from post no {all_posts.total_index}, using {path_to_resume} ')             

                for post in all_posts:

                    now = datetime.datetime.now()
                    
                    # sleeping from 12 a.m to 7 a.m
                    while (now.hour >= night_sleep and now.hour < morning_wake) and response:
                        print(f'hour = {now.hour}')
                        now = datetime.datetime.now()
                        start = time.time()
                        print(f'sleeping for {night_sleep_time} minute . . .')
                        time.sleep(night_sleep_time*60)
                    
                    end = time.time()
                    elapsed_time_in_min = (end - start)/60
                    print(f'elapsed_time is {elapsed_time_in_min:.2f} minute')
                    print(f'counter = {counter}\n')

                    # sleeping if more than max_download pictures is downloaded in less than min_time miutes.
                    if counter > max_download and response:
                        if elapsed_time_in_min < min_time:
                            counter = 0
                            sleep = random.uniform(down_sleep_min*60,down_sleep_max*60)
                            print(f'sleep for {sleep/60:.2f} minute . . .')
                            time.sleep(sleep)
                            start = time.time()
                        else:
                            print('reseting the counter and chronometer')
                            counter = 0
                            start = time.time()

                    #probability(specified in config.yaml) to sleep and change proxy if the previous response was successful (indicate the file does not exists). 
                    if random.randint(1,int(1/random_probability)) == 1 and response:     
                        i = self.change_proxy(proxies, i)
                        if login_users_path:
                                user, password= self.change_user(self.users, self.login_users)
                                self._loader_login(user, password)
                        sleep_time =random.uniform(random_sleep_time_start, random_sleep_time_end)
                        print(f'sleeping for {sleep_time:.2f} seconds')
                        time.sleep(sleep_time)
                    
                    # downloading the post
                    response = self.loader.download_post(post, target=self.profile.username)
                    counter += 1
                    print(f'{all_posts.total_index}/{all_posts.count} posts downloaded.')

            # exception if any error happened while downloading the post
            except Exception:
                print('\nException occured, saving the structure to file...')
                save_structure_to_file(all_posts.freeze(), str(path_to_resume))
                i = self.change_proxy(proxies, i)

        # removing resume_information.json when the download is finished.
        self.remove_resume(str(path_to_resume))
        print(f'########################{profile_name} profile downlaod finished.########################')

    # function to change the username
    @staticmethod
    def change_user(users, login_users):
        user = users[random.randint(0, len(users)-1)]
        password = login_users[user]
        print(f'using "{user}" for logging into the instagram...')

        return user, password
       
    # function to change the proxy
    @staticmethod
    def change_proxy(proxies, i):
        # proxy = proxies[random.randint(0, len(proxies)-1)]
        if i> (len(proxies)-1):
            i = 0
        proxy = proxies[i]
        os.environ['HTTPS_PROXY'] = os.environ['https_proxy'] = f'{proxy}'
        print(f"\nSwitching proxy and using '{proxy}' for connecting...")
        os.system('curl https://httpbin.org/ip')

        return i+1

    # function to remove resume file
    @staticmethod
    def remove_resume(path_to_resume):
        print('removing resume_information.json...')
        try:
            os.remove(path_to_resume)
            print('resume_information.json removed')
        except Exception:
            print('resume_information.json not exists')   
    
    # function to iterate in profiles and download one by one
    @ex.capture
    def run(self, profiles_path):
        with open(profiles_path) as p:
            profiles = p.read().split('\n')
            try:
                profiles.remove('')
            except Exception:
                pass
        for profile_name in profiles:
            self.download(profile_name)

@ex.config
def get_config():
    config_path = str(Path(__file__).resolve().parents[0].joinpath('configs', 'config.yaml'))
    ex.add_config(config_path)

@ex.automain
def main(_config,_run):
    downloader = DownloadProfile(_config)
    downloader.run()


## TO DO:
# putting more random behaviour such as putting comments and likes
# connecting users to specific proxies
# InstaloaderContext
# start using the logging