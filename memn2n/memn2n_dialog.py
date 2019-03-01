from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
from datetime import datetime
import copy
import pdb
import random
def zero_nil_slot(t, name=None):
    """

    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope( name, "zero_nil_slot",[t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        # z = tf.zeros([1, s])
        return tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], 0, name=name)


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope( name, "add_gradient_noise",[t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

def find_lcseque(s1, s2):
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    # pdb.set_trace()
    # print (np.array(d))
    s = []
    while m[p1][p2]:
        c = d[p1][p2]
        if c == 'ok':
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':
            p2 -= 1
        if c == 'up':
            p1 -= 1
    s.reverse()
    return s


class MemN2NDialog(object):
    """End-To-End Memory Network."""

    def __init__(self, batch_size, vocab_size, candidates_size, sentence_size, embedding_size,
                 candidates_vec,
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 session=tf.Session(),
                 name='MemN2N',
                 task_id=1,introspection_times=None,my_embedding=None):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            candidates_size: The size of candidates

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            candidates_vec: The numpy array of candidates encoding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._candidates_size = candidates_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._name = name
        self._candidates = candidates_vec

        self._my_embedding = my_embedding
        self._build_inputs()
        self._build_vars()
        self._intro_times=introspection_times
        # define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = "%s_%s_%s_%s/" % ('task',
                                          str(task_id), 'summary_output', timestamp)

        # cross entropy
        # (batch_size, candidates_size)
        logits = self._inference(self._stories, self._queries)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self._answers, name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(
            cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = cross_entropy_sum

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        pdb.set_trace()
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                          for g, v in grads_and_vars]
        # grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(
            nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(
            predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        self.graph_output = self.loss_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._stories = tf.placeholder(
            tf.int32, [None, None, self._sentence_size], name="stories")
        self._queries = tf.placeholder(
            tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answers")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])

            if self._my_embedding is not None:
                initial=tf.constant_initializer(value=self._my_embedding,dtype=tf.float32)
                pdb.set_trace()
                # trained_emb=tf.get_variable('embedding_word', shape=[self._vocab_size, self._embedding_size],
                #                 initializer=initial, trainable=True)
                # A = tf.concat([nil_word_slot,initial ], 0)
                self.A = tf.get_variable(name="A",shape=[self._vocab_size, self._embedding_size],initializer=initial)
                self.H = tf.Variable(self._init(
                    [self._embedding_size, self._embedding_size]), name="H")
                W = tf.concat([nil_word_slot, tf.get_variable('trained_embedding_W', shape=[self._vocab_size, self._embedding_size],initializer=initial, trainable=True)], 0)
                self.W = tf.Variable(W, name="W")
            else:
                A = tf.concat([nil_word_slot, self._init(
                    [self._vocab_size - 1, self._embedding_size])], 0)
                self.A = tf.Variable(A, name="A")
                self.H = tf.Variable(self._init(
                    [self._embedding_size, self._embedding_size]), name="H")
                W = tf.concat([nil_word_slot, self._init(
                    [self._vocab_size - 1, self._embedding_size])], 0)
                self.W = tf.Variable(W, name="W")
            # self.W = tf.Variable(self._init([self._vocab_size, self._embedding_size]), name="W")
        self._nil_vars = set([self.A.name, self.W.name])

    def _inference(self, stories, queries):
        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(self.A, queries)
            u_0 = tf.reduce_sum(q_emb, 1)
            u = [u_0]
            for _ in range(self._hops):
                m_emb = tf.nn.embedding_lookup(self.A, stories)
                m = tf.reduce_sum(m_emb, 2)
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(m, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = tf.matmul(u[-1], self.H) + o_k
                # u_k=u[-1]+tf.matmul(o_k,self.H)
                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)
            candidates_emb = tf.nn.embedding_lookup(self.W, self._candidates) #candidates.shape=[4212,9]
            candidates_emb_sum = tf.reduce_sum(candidates_emb, 1)#candidates_emb.shape=[4212,9,20]
            # pdb.set_trace()
            return tf.matmul(u_k, tf.transpose(candidates_emb_sum))#candidates_emb_sum.shape=[4212,20]
            # logits=tf.matmul(u_k, self.W)
            # return
            # tf.transpose(tf.sparse_tensor_dense_matmul(self._candidates,tf.transpose(logits)))

    def batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories,
                     self._queries: queries, self._answers: answers}
        # pdb.set_trace()
        # try:
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        # except:
        #     for id,story in enumerate(stories):
        #         print (id,':',len(story))
        #     for id, story in enumerate(queries):
        #         print (id, ':', len(story))
        #     pdb.set_trace()
        return loss

    def predict(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)


    def simulate_query(self, test_stories, test_queries, test_tags, train_data, word_idx, train_word_set,idx_word):
        # pdb.set_trace()
        s = train_data[0]
        q = train_data[1]
        a = train_data[2]
        tags_train = train_data[3]
        tags_test = test_tags
        # pdb.set_trace()
        name_map_ = self.entities_map(tags_test, tags_train, s, test_stories, train_word_set,idx_word)
        # pdb.set_trace()
        print('vocab len:', len(name_map_))
        print ('name_map_:',name_map_)
        for key,values in name_map_.items():
            print ('key:',idx_word[key],':')
            for value in values:
                print(idx_word[value])

        for s_e in range(self._intro_times):
            name_map = {}
        # choice_count={}
            for test_entity, train_entities in name_map_.items():
        #     choice_count[test_entity]=len(train_entities)
        # while len(choice_count)>0:
            # pdb.set_trace()
            #test_entity=sorted(choice_count, key=lambda x: choice_count[x])[-1]
            # choice_count.pop(test_entity)
        # for test_entity, in name_map_.keys():
        #     count_map = {}
        #     for train_entity in name_map_[test_entity]:
        #         if train_entity in count_map.keys():
        #             count_map[train_entity]+=1
        #         else:
        #             count_map[train_entity]=1

            # pdb.set_trace()
            # print('last count_map:', count_map)
            # while len(count_map)>0:
            #     vot_result = sorted(count_map, key=lambda x: count_map[x])[-1]
                random_result=random.sample(train_entities,1)
                # if random_result not in name_map.values() and test_entity!=random_result:
                name_map[test_entity] = random_result[0]
                #     break
                # else:
                #     count_map.pop(vot_result)

        # pdb.set_trace()
        # if not len(name_map) == len(name_map_): pdb.set_trace()
            name_map = {value: key for key, value in name_map.items()}
            # for key,value in name_map.items():
            #     print (idx_word[key],idx_word[value])
        # pdb.set_trace()
        # idx_word={value:key for key,value in word_idx.items()}
        # for key,value in name_map.items():
        #     print(idx_word[key],idx_word[value],'\n')
        # pdb.set_trace()
        # print('new name_map:', name_map)
        # for query in test_queries:
        #     a_list=test_entities
        #     b_list=query
        #     cross=list((set(a_list).union(set(b_list))) ^ (set(a_list) ^ set(b_list)))
        #     if len(cross)>0:
        # pdb.set_trace()
        # print('simulate querying...')

        # losses = 0

            losses = self.simulate_train(name_map, s, q, a)
            print('The %d th simulation loss:%f' % (s_e, losses))

    def entities_map(self, tags_test, tags_train, train_stories, test_stories, train_set,idx_word):
        name_map = {}
        # samples=[]
        def similar_sample(tags_test_sent_, tags_train_, position):
            similar_sample_index = []
            longest_len = 0
            for idx_story, tags_story in enumerate(tags_train_):
                for idx_sent, tags_sents in enumerate(tags_story):
                    length = len(find_lcseque(tags_test_sent_, tags_sents))
                    if length>longest_len and len(tags_sents) > position:
                        longest_len = length
                        similar_sample_index=[]
                        similar_sample_index.append([idx_story, idx_sent])
                    if length == longest_len and len(tags_sents) > position:
                        similar_sample_index.append([idx_story, idx_sent])
                # if len(similar_sample_index)>500:
                #     break
            return similar_sample_index

        def new_words_position(sent, train_set,idx_word):
            new_words_p = []
            new_word = []
            # token = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
            for idx, word in enumerate(sent):
                if word not in train_set and not word == 0 and '_' not in idx_word[word]:
                    new_words_p.append(idx)
                    new_word.append(word)
                # pdb.set_trace()
            return new_words_p, new_word

        for idx_story, story_test in enumerate(test_stories):
            # print('test number:', idx_story)
            for idx_sents, sents in enumerate(story_test):
                # pdb.set_trace()
                position_list_, new_words_ = new_words_position(sents[:-1], train_set,idx_word)
                # pdb.set_trace()
                position_list,new_words=[],[]
                for idx,words in enumerate(new_words_):
                    if words not in name_map.keys():
                        new_words.append(words)
                        position_list.append(position_list_[idx])
                # print (recognise,new_words,name_map)
                if len(position_list) > 0:
                    for position in position_list:
                        similar_smaple_in_train_positions = similar_sample(tags_test[idx_story][idx_sents], tags_train, position)
                        # pdb.set_trace()
                        for train_position in similar_smaple_in_train_positions:
                          try:

                            if tags_train[train_position[0]][train_position[1]][position] == \
                                    tags_test[idx_story][idx_sents][position]:
                                # pdb.set_trace()
                                value = train_stories[train_position[0]][train_position[1]][position]

                            else:
                                test_pos=tags_test[idx_story][idx_sents][position]
                                train_sents_pos=tags_train[train_position[0]][train_position[1]]
                                if test_pos in train_sents_pos:
                                    near_position=train_sents_pos.index(test_pos)
                                    value=train_stories[train_position[0]][train_position[1]][near_position]
                                else:
                                    continue
                            if '_' in idx_word[value] or '#' in idx_word[value]:
                                continue
                            if sents[position] not in name_map.keys():
                                name_map[sents[position]] = [value]
                            elif value not in name_map[sents[position]]:
                                name_map[sents[position]].append(value)
                          except:
                                pdb.set_trace()

        return name_map

    def simulate_train(self, name_map, story, query, answer):
        stories, queries, answers = [], [], []
        # for key,value in name_map.items():
        #     name_map_temp={value:key}
        flag = False
        for i in range(len(query)):
            s = copy.copy(story[i])
            q = copy.copy(query[i])
            a = copy.copy(answer[i])
            for no, id in enumerate(q):
                if id in name_map.keys():
                    q[no] = name_map[id]
                    flag = True
            for _no, sent in enumerate(s):
                for no_, id in enumerate(sent):
                    if id in name_map.keys():
                        sent[no_] = name_map[id]
                        flag = True
            # pdb.set_trace()

            a_new =int(a)
            if a_new in name_map.keys():
                a_new=name_map[a_new]
                flag = True
            # pdb.set_trace()
            if flag:
                stories.append(s)
                queries.append(q)
                answers.append(np.array(a_new))
                flag = False

        if len(queries) <= 0: pdb.set_trace()
        total_cost = 0.0
        # print('simulate samples number:', len(stories))
        if len(queries) > self._batch_size:
            batches = zip(range(0, len(queries) - self._batch_size, self._batch_size), range(self._batch_size, len(queries), self._batch_size))
            batches = [(start, end) for start, end in batches]
            np.random.shuffle(batches)
            # pdb.set_trace()
            for start, end in batches:
                s = stories[start:end]
                q = queries[start:end]
                a = answers[start:end]
                # pdb.set_trace()
                cost_t = self.batch_fit(s, q, a)
                total_cost += cost_t
        else:
            total_cost = self.batch_fit(stories, queries, answers)
        return total_cost
