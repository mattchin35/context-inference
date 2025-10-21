from one.api import ONE
from icecream import ic
import brainbox.behavior.wheel as wh

# one = ONE(base_url='https://openalyx.internationalbrainlab.org')
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

sessions = one.search()
print(sessions[0],)

# Find an example session with trials data
eid = one.search(project='brainwide', datasets='spikes.times.npy', subject='UCLA033')
ic(eid)
# eid = '6be21156-33b0-4f70-9a0f-65b3e3cd6d4a'
# List datasets associated with a session, in the alf collection
datasets = one.list_datasets(eid[0], collection='alf*')
ic(datasets)

# probe_insertions = one.load_dataset(eid, 'probes.description')
#
# print(f'N probes = {len(probe_insertions)}')
# pprint(probe_insertions[0])

# access specific items
# trials = one.load_object(eid, 'trials')
# wheel = one.load_object(eid, 'wheel')

# Download all data in alf collection
files = one.load_collection(eid, 'alf*', download_only=True)
# Show where files have been downloaded to
print(f'Files downloaded to {files[1].parent}')

