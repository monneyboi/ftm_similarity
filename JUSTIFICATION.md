**1. Explain your choices for preprocessing the entities. Explain how you might improve this, given more time and a greater variety and amount of entities.**

I'm pre-computing composite properties (only "name" in this case). Depending on the strategy, it could make sense to generate descriptive properties as well (for example based on relations).

I think we could improve upon this tactic by taking into account the relations of entities in our dataset, as there is only so much similar properties can tell you.

**2. Explain your choices for embedding the entities and your choice of vector store. Explain how you might improve this, given more time and a greater variety and amount of entities.**

In this demo, everything is dumped into pickle files, in the real world we'd have to use a serious vector database.

I started of by thinking of using the relations in the FTM data to compare entities, as i think that in real world situations, these will be very important. The right way to do this would be with some graph to vector kind of technique, but after some prompting i ended up creating stories for entities, so i could use a single embedding score and keep things as simple as possible, while allowing for adding the semantics of the relations to the embedded information.

After discussing my reasoning with the interviewer, it became clear that i would be working with sythetic data and that the entities have no relations. This made the story approach less viable, so i switched my approach to embedding all properties seperately, and on searching, calculating a weighted average. A simpler approach might be to still create a single embedding, with important properties included multiple times, though i did not test this.

Given more time i would have used a dataset of known similar entities to train the weights.

As stated before, i think using the relations in the dataset could be very interesting.

**3. Explain _what you would need to know_ in order to turn this product into a service that embeds, stores, and matches millions of entities, and serves these matches RESTfully to other applications. This is your turn to ask questions about requirements, tech stack, and data. Ask the _most important, high-level_ questions (no more than 5) that could help you come up with a specification. Explain why these questions are important, what different answers could mean, but do not actually attempt to write a specification.**

- What would be the expected queries this service would have to respond to, and what would be the performance requirements.
- How dynamic is the data? Does it change often?
- What is the desired matching behaviour and how acceptable are false positives / negatives? Do these change per query?
- What's the expected dataset size and what's the budget for compute/storage.
- What level of auditability is needed? Do users need to understand why entities matched?
