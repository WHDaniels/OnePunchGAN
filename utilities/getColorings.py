"""
Not relevant for final approach...
"""

from psaw import PushshiftAPI
import datetime as dt
import praw.models
import pickle
import time


def collectIDs(startDate=dt.datetime(2021, 7, 9), subreddit='onepunchman', interval=250000):
    """
    Collects IDs of every reddit post on the specified subreddit.
    :param startDate: Day you want to start collecting IDs from.
    :param subreddit: Subreddit to collect IDs from.
    :param interval: Epoch time interval to collect each iteration.
    :return: None
    """

    startTime = int(startDate.timestamp())
    endTime = int(dt.datetime(2021, 5, 20).timestamp())
    lastChunkGap = False
    gapList = list()
    idList = list()

    i = 1
    while startTime > endTime:
        stopTime = startTime - interval
        print(dt.datetime.fromtimestamp(startTime))

        submissionChunk = list(api.search_submissions(after=stopTime, before=startTime,
                                                      subreddit=subreddit, filter=['id'], limit=None))

        print("Chunk length:", len(submissionChunk))
        print("Chunk:", submissionChunk, "\n")

        # Check and handle data gaps ------------------
        if len(submissionChunk) == 0 and not lastChunkGap:
            gapStart = startTime
            lastChunkGap = True

        elif len(submissionChunk) > 0 and lastChunkGap:
            gapEnd = stopTime
            lastChunkGap = False

            # gap closed, so keep track of gap
            gapList.append((dt.datetime.fromtimestamp(gapStart), dt.datetime.fromtimestamp(gapEnd)))

        # --------------------------------------------

        for entry in submissionChunk:
            idList.append(entry.id)

        startTime = stopTime
        i += 1

    pickle.dump(idList, open('data//idList3.p', 'wb'))
    pickle.dump(gapList, open('data//gapList3.p', 'wb'))


def seeGaps():
    gapList = pickle.load(open('data//gapList3.p', 'rb'))
    for x in gapList:
        print(x)


def makeIDLists(idFullList):
    OC_list = list()
    artList = list()
    coloredList = list()

    for n, post in enumerate(idFullList):
        start = time.perf_counter()

        print("\nPost", n)
        submission = praw.models.Submission(reddit, id=idFullList[n])

        if 'colo' in submission.title.lower():
            print("Coloring found")
            coloredList.append(idFullList[n])
            pickle.dump(coloredList, open('data//list3//coloredList3.p', 'wb'))

        elif submission.link_flair_text is not None:
            if 'colo' in submission.link_flair_text.lower():
                print("Coloring found")
                coloredList.append(idFullList[n])
                pickle.dump(coloredList, open('data//list3//coloredList3.p', 'wb'))
            elif 'art' in submission.link_flair_text.lower():
                print("Art found")
                artList.append(idFullList[n])
                pickle.dump(artList, open('data//list3//artList3.p', 'wb'))

        elif submission.is_original_content:
            print("OC found")
            OC_list.append(idFullList[n])
            pickle.dump(OC_list, open('data//list3//OC_list3.p', 'wb'))

        time.sleep(0.8)

        end = time.perf_counter()
        print(reddit.auth.limits)
        print("Time used:", end - start)


def getID_after(ID, idList):
    for n, entry in enumerate(idList):
        if entry == ID:
            for post in idList[n + 50:n + 500]:
                print(post)


if __name__ == '__main__':
    api = PushshiftAPI()

    # get credentials from ini
    reddit = praw.Reddit()
    authUser = reddit.user.me()

    collectIDs()
    idFullList = pickle.load(open('data//idList3.p', 'rb'))
    makeIDLists(idFullList)

    # idColoredList = pickle.load(open('data//coloredList.p', 'rb'))
    # seeGaps()
