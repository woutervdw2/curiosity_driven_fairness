import train_agent

env_params = attention_allocation.Params(
    n_locations = 6,
    prior_incident_counts =  (600, 500, 400, 300, 200, 100),
    incident_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    n_attention_units = 4,
    miss_incident_prob = (0., 0., 0., 0., 0., 0.),
    extra_incident_prob = (0., 0., 0., 0., 0., 0.),
    dynamic_rate = 0.1
)

MODELS_TO_TRAIN = ['visit_count', 'scalar', 'UCB']
LEARNING_STEPS = 10000

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')+)'/parms_test/'

if not os.path.exists(PATH):
    os.makedirs(PATH)



#Define parameters to optimize
lr = [0.1, 0.01, 0.001, 0.0001]
gamma = [0.99, 0.9, 0.8, 0.7]
clip_range = [0.2, 0.1, 0.05, 0.01]
n_steps = [32, 64, 128, 256, 512]
model = 'UCB'

#Define parameter combinations
param_combinations = list(itertools.product(lr, gamma, clip_range, n_steps))

#Define csv file to save results
filename = PATH + 'attention_results.csv'
fieldnames = ['index', 'lr', 'gamma', 'clip_range', 'n_steps', 'model', 'found_incidents']

#Check if file already exists
if os.path.exists(filename):
    with open(filename, mode='r') as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        existing_combinations = set((float(row['lr']), float(row['gamma']), float(row['clip_range']),
                                      int(row['n_steps']), row['model']) for row in reader)
else:
    # If the file doesn't exist, create an empty set of existing parameter combinations
    existing_combinations = set()


#Run parameter optimization save results to csv file 
with open(filename, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not os.path.isfile(filename):
        print('File does not exist, creating new file')
        writer.writeheader()

    for i, params in enumerate(tqdm(param_combinations)):
        lr, gamma, clip_range, n_steps = params

        if (lr, gamma, clip_range, n_steps, model) in existing_combinations:
            continue
        
        env = train_agent.init_env(env_params, rewards=model, test=False)
        test_env = train_agent.init_env(env_params, rewards=model, test=True)
        
        agent = train_agent.train_agent(env, PATH, rewards=model, learning_rate=lr, n_steps=n_steps,
                                        gamma=gamma, clip_range=clip_range, verbose=0, learning_steps=LEARNING_STEPS, save=False)
        
        #Write evaluation on test environmetn
        
        result = np.mean(test_results['bank_cash'])
        writer.writerow({'index': i, 'lr': lr, 'gamma': gamma, 'clip_range': clip_range, 
                            'n_steps': n_steps, 'model': model, 'bank_cash': result})
        f.flush()
