from instaloader import Instaloader, Profile
import random
import time


class DownloadProfile:

    def __init__(
                self, 
                profile_name,
                username,
                password,
                download_dir,
                download_comments=False,
                download_videos=False,
                save_metadata = False,
                download_geotags = False,
                post_metadata_txt_pattern=''
                ):
        '''
        Download post from a profile
        :param profile_name: str, name of the profile to download
        :param username: str, username to login
        :param password: str, pasword to login
        :param download_dir: str, directory to download the data
        ** other Instaloader keywords **
        '''
        self.profile_name = profile_name
        self.username = username
        self.password = password
        self.download_dir = download_dir
        self.download_comments = download_comments
        self.download_videos = download_videos
        self.save_metadata = save_metadata
        self.download_geotags = download_geotags
        self.post_metadata_txt_pattern = post_metadata_txt_pattern
        self._make_loader()
        self._make_profile()

    def _make_loader(self):

        self.loader = Instaloader(
                                download_videos=self.download_videos,
                                download_comments=self.download_comments,
                                download_geotags=self.download_geotags,
                                save_metadata=self.save_metadata,
                                dirname_pattern = self.download_dir,
                                post_metadata_txt_pattern = self.post_metadata_txt_pattern,
                                max_connection_attempts = 0 #infinit times for json request problem
                                )
        self.loader.login(self.username,self.password)

    def _make_profile(self):

        self.profile =  Profile.from_username(self.loader.context, self.profile_name)

    def download(self):

        response = False
        for post in self.profile.get_posts():

            if random.randint(0, 1) and response:
                sleep_time = random.uniform(4, 7)
                print(f'sleeping for {sleep_time}')
                time.sleep(sleep_time)
            response = self.loader.download_post(post, target=self.profile.username)

if __name__ == '__main__':
    cities = {
    # "tehran": ['tehran.zoom', 'explore_tehran', 'tehranazdoor', 'tehran', 'eye_on_tehran'],
    "dubai": ['explore.dubai_', 'dubai', 'visit.dubai'],
    "istanbul": ['istanbul', 'istanbull.hayali'],
    "dushanbe": ['visitqatar', 'doha'],
    "muscat": ['experienceoman', 'oman_tourism_love'],
    "baku": ['visitazerbaijan', 'experienceazerbaijan'],
    "doha": ['dushanbe']
    }
    
    for city , profile_list in cities.items():
        download_dir= f'/data/reza/datasets/place/{city}'
        for profile_name in profile_list:
            profile = DownloadProfile(profile_name, username='cityandimages', password='urban_article2021', download_dir=download_dir,
                                    download_comments=False, download_videos=False, save_metadata=False, download_geotags=False, 
                                    post_metadata_txt_pattern='')
            profile.download()