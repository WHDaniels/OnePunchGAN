import pickle
import praw.models
from urllib.error import HTTPError
from urllib.error import URLError
import urllib.request as requests

imgExtensionList = ['.jpg', '.png', '.jpeg', '.bmp']
excludeList = ['youtube', 'gfycat', 'instagram', 'ustream']
# downloadFolder = 'E://OPM Colorings//downloads//'
# downloadFolder = 'E://OPM Colorings//art_downloads//'
failList = list()


def download(idList, downloadFolder):
    for n, post_id in enumerate(reversed(idList)):
        submission = praw.models.Submission(reddit, id=post_id)
        link = submission.url

        print("Post", n, link)

        exclusionFlag = False
        for exclusion in excludeList:
            if exclusion in link:
                exclusionFlag = True

        if exclusionFlag:
            print("Exclusion, skipping...")
            continue

        if submission.is_self is True:
            print("Text only, skipping...")
            continue

        extension = '.' + link.split('.')[-1]
        if extension not in imgExtensionList:
            extension = '.jpg'

        try:
            if 'imgur' in link:

                if '/a/' in link:
                    requests.urlretrieve(link + '/zip', downloadFolder + str(n) + '.zip')
                elif '.png' != extension:
                    requests.urlretrieve(link + '.png', downloadFolder + str(n) + extension)
                else:
                    requests.urlretrieve(link, downloadFolder + str(n) + extension)
            else:
                requests.urlretrieve(link, downloadFolder + str(n) + extension)

        except HTTPError as http_e:
            print("HTTP Error", http_e.code, "occurred, skipping...")
        except URLError as url_e:
            print("URL Error", url_e, "occurred, skipping...")
        except Exception as e:
            print("Error occurred (" + str(e) + "), skipping...")


def cleanDownloads():
    print("Cleaning...")
    # failed


if __name__ == '__main__':
    coloredIDs = pickle.load(open('data//list3//coloredList3.p', 'rb'))
    artIDs = pickle.load(open('data//list3//artList3.p', 'rb'))

    reddit = praw.Reddit()
    authUser = reddit.user.me()

    print(len(coloredIDs))
    download(coloredIDs, "E://OPM//OPM Colorings//downloads3//")
    print(len(artIDs))
    download(artIDs, "E://OPM//OPM Colorings//art_downloads3//")
