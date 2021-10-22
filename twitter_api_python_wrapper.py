# credentials are in a yaml file
# credential handling
from searchtweets import gen_rule_payload, load_credentials, collect_results
# Enterprise setup
enterprise_search_args = load_credentials(filename="./search_tweets_creds_example.yaml",
                                          yaml_key="search_tweets_enterprise_example",
                                          env_overwrite=False)
# (returns a json object with keys: 'username', 'passwords', 'endpoint')

# Premium setup
premium_search_args = load_credentials(filename = "./search_tweets_creds_example.yaml",
                                       yaml_key="search_tweets_premium_example",
                                       env_overwrite=False)
# (returns a json object with keys: 'bearer_token', 'endpoint', 'extra_headers_dict')

# formats search API rules into valid json queries
rule = gen_rule_payload("covid-19",
                        from_date="2021-09-01", #UTC at 00:00
                        to_date="2021-10-01",#UTC at 00:00
                        results_per_call=500)
print(rule)

tweets = collect_results(rule, # a valid PowerTrack rule
                         max_results=100,
                         result_stream_args=enterprise_search_args) 
[print(tweet.all_text, end='\n\n') for tweet in tweets[0:10]];
[print(tweet.created_at_datetime) for tweet in tweets[0:10]];
[print(tweet.generator.get("name")) for tweet in tweets[0:10]];