# System Design

Steps:
1. Requirements engineering
2. Capacity estimation
3. Data modeling
4. API design
5. System design
6. Design discussion

## 1. Requirements engineering
Functional and non-functional requirements.
Scale of the system.

### 1.1. Functional requirements
What the system should do.
- Which problem does the system solve
- Which features are essential to solve these problems

Core features: bare minimum required to solve the user's problem.
Support features: features that help the user solve their problem more conveniently.

### 1.2. Non-functional requirements
What the system should handle.

>**Interview tips**
> - Identify the nonfunctional requirements the interviewer cares most about
> - Show awareness of system attributes, trade-offs and user experience

System analysis:
- Read or write heavy?
  - Impacts requirements for database scaling, redundancy of services, effectiveness of caching.
  - System priorities - is it worse if the system fails to read or to write?
- Monolithic or distributed architecture?
  - Scale is the key consideration. Large scale applications must be distributed, smaller scale can be single server.
- Data availability vs consistency
  - CAP trilemma - A system can have only two of: Consistency, Availability, Partition tolerance
  - Choice determines the impact of a network failure

Non-functional requirements:
- Availability - How long is the system up and running per year?
  - Availability of a system is the **product** of its components' availability
  - Reliability, redundancy, fault tolerance
- Consistency - Data appears the same on all nodes regardless of the user and location.
  - Linearizability
- Scalability - Ability of a system to handle fast-growing user base and temporary load spikes
  - Vertical scaling - single server that we add more CPU/RAM to
    - Pros: fast inter-process communication, data consistency
    - Cons: single point of failure, hardware limits on maximum scale
  - Horizontal scaling - cluster of servers in parallel
    - Pros: redundancy improves availability, linear cost of scaling
    - Cons: complex architecture, data inconsistency
- Latency - Amount of time taken for a single message to be delivered
  - Experienced latency = network latency + system latency
  - Database and data model choices affect latency
  - Caching can be effective
- Compatibility - Ability of a system to operate seamlessly with other software, hardware and systems
- Security
  - Basic security measures: TLS encrypted network traffic, API keys for rate limiting

Non-functional requirements depend heavily on expected scale. 
The following help quantify expected scale, and relate to capacity estimation in the next step.
- Daily active users (DAU)
- Peak active users
- Interactions per user - including read/write ratio
- Request size
- Replication factor - determines the storage requirement (typically 3x)

## 2. Capacity Estimation

Requests (requests per second), bandwidth (requests * message size), storage


## 3. Data modeling
Entities, attributes, relations.


## 4. API design
Endpoints with their parameters, response and status codes.


## 5. System design
System components: 
- Representation layer
  - Web app
  - Mobile app
  - Console
- Service (and load balancer)
- Data store
  - Relational database
    - Pros: linked tables reduce duplicate data; flexible queries
  - Key-value store
    - Useful for lots of small continuous reads and writes
    - Pros: Fast access, lightweight, highly scalable
    - Cons: Cannot query by value, no standard query language so data access in written in the application layer
            no data normalisation

Scaling: 
- Scale services by having multiple instances of the service with a load balancer in front
- Scale database with federation (functional partitioning) - split up data by function

System diagrams: arrows point in direction of user flow (not data flow)


### Databases
#### Relational databases
4 main operations (CRUD): Create, Read, Update, Delete.

Transactions - group several operations together
- Either the entire transaction succeeds or it fails and rolls back to the initial state. Commit or abort on error.
- Without transactions, if an individual operation fails, it can be complex to unwind the related transactions to return to the initial state.
- Error handling is simpler as manual rollback of operations is not required.
- Transactions provide guarantees so we can reason about the database state before and after the transaction.

Transactions enforce "ACID" guarantees:
- Atomicity -    Transactions cannot be broken down into smaller parts. 
- Consistency -  All data points within the database must align to be properly read and accepted. Raise a consistency error if this is not the case.
                 Database consistency is different from system consistency which states the data should be the same across all nodes.
- Isolation -    Concurrently executing transactions are isolated from each other.
                 Avoids race conditions if multiple clients are accessing the same database record.
- Durability -   Once a transaction is committed to the database, it is permanently preserved.
                 Backups and transaction logs can restore committed transactions in the case of a failure. 

Relational databases are a good fit when:
- Data is well-structured
- Use case requires complex querying
- Data consistency is important

Limitations
- Horizontal scaling is complex
- Distributed databases are more complex to keep transaction guarantees
  - Two phase commit protocol (2PC):
    1. Prepare - Ask each node if it;s able to promise to carry out the transaction
    2. Commit - Block the nodes and do the commit
  - Blocking causes problems on distributed systems where it may cause unexpected consequences if the database is unavailable for this short time. 

#### Non-relational databases
Examples of non-relational databases:
- Key-value store
- Document store
- Wide-column store
- Graph store

These vary a lot, and in general NoSQL simply means "no ACID guarantees".

Transactions were seen as the enemy of scalability, so needed to be abandoned completely for performance and scalability.
ACID enforces consistency for relational databases; for non-relational databases there is eventual consistency.

BASE:
- Basically Available -  Always possible to read and write data even though it may not be consistent.
                         E.g. reads may not reflect latest changes, writes may not be persisted.
- Soft state -           Lack of consistency guarantees mean data state may change without any interactions with the application
                         as the database reaches eventual consistency. 
- Eventual consistency - Data will eventually become consistent once inputs stops.

Benefits:
- Without atomicity constraint, overheads like 2PC are not required
- Without consistency constraint, horizontal scaling is trivial
- Without isolation constraint, no blocking is required which improves availability.

Non-relational databases are a good fit when:
- Large data volume that isn't tabular
- High availability requirement
- Lack of consistency across nodes is acceptable

Limitations:
- Consistency is necessary for some use cases
- Lack of standardisation


## 6. Design discussion



## 7. System design examples and discussion
### 7.1. Todo App
An app that lets a user add and delete items from a todo list.
- Representation layer is a web app
- Microservices handle the todo CRUD operations and the user details separately
  - Each service is scaled horizontally, so a load balancer is placed in front of it
- Relational database stores data
  - There are two databases, one per service, which is an example of functional partitioning. 
    This means todo service I/O does not interfere with user service I/O and vice versa.

![img.png](../_images/system_design/design_todo_app.png)

### 7.2. URL shorteners
Take an input URL and return a unique, shortened URL that redirects to the original.

#### Encoding deep-dive
There are two parameters we can vary:
1. Length of the key - should be limited so the URL is short enough
2. Range of allowed characters - should be URL-safe

Using Base-64 encoding as the character range, then a 6-digit key length gives enough unique URLs;
64^6 = 68 billion

Encoding options:
1. MD5 hash: not collision resistant and too long
2. Encoded counter, each URL is just an index int, and its value is the base-64 encoding of that int:
   length is not fixed and increases over time
3. Key range, generate all unique keys beforehand: no collisions, but storage intensive
4. Key range modified, generate 5-digit keys beforehand, then add the 6th digit on the fly.

Planned system architecture:
- Pre-generate all 5 digit keys (1 billion entries rather than 64 billion for 6 digits)
- System retrieves 5-difit keys from database
- System appends 1 out of 64 characters
- The system is extendable because if we run out of keys we can append 2 more characters rather than 1

#### Mock interview
System analysis:
- System is read heavy - Once a short URL is created it will be read multiple times
- Distributed system as this has to scale
- Availability > Consistency - not so much of a concern if a link isn't available to all users at the same time, 
  but they must be unique and lead to their destination.


**Requirements engineering:-**
Core feature:
- A user can input a URL of arbitrary length and receive a unique short URL of fixed size.
- A user can navigate to a short link and be redirected to the original URL.

Support features:
- A user can see their link history of all created short URLs.
- Lifecycle policy - links should expire after a default time span.

Non-functional requirements:
- Availability - the system should be available 99% of the time.
- Scalability - the system should support billions of short URLs, and thousands of concurrent users.
- Latency - the system should return redirect from a short URL to the original in under 1 second.

Questions to capture scale:
- Daily active users, and how often do users interact per day
  - 100 million, 1 interaction per day
- Peak active users - are there events that lead to traffic spikes?
  - No spikes
- Read/write ratio
  - 10 to 1
- Request size - how long are the originals URLs typically
  - 200 characters => 200 Bytes
- Replication factor
  - 1x - ignore replication

**Capacity estimation:-**
- Requests per second 
  - Reads per second = DAU * interactions per day / seconds in day = 10^8 * 1 / 10^5 = 1000 reads/s
  - Writes per second = Reads per second / read write ratio = 1000 / 10 = 100 writes/s
  - Requests per second = Reads per second + Writes per second = 1000 + 100 = 1100 requests/s
  - No peak loads to consider
- Bandwidth 
  - Bandwidth = Requests per second * Message size
  - Read bandwidth = 1000 * 200 Bytes = 200kB/s
  - Write bandwidth = 100 * 200 Bytes = 20 kB/s
- Storage
  - Storage per year = Write bandwidth * seconds per year * Replication factor = 20000 Bytes * (3600*24*365) * 1 = 630 GB


**Data model:-**
Identify entities, attributes and relationships.

Entities and attributes: 
- Links: Key (used to create short URL, Original URL, Expiry date
- Users: UserID, Links
- Key ranges: Key range, In use (bool)

Relationships:
- Users own Links
- Links belong to Key ranges

Data stores:
- Users
  - User data is typically relational and we rarely want it all returned at once. 
  - Consistency is important as we want the user to have the same experience regardless of which server handles their log in, and don't wat userIds to clash.
  - Relational database.
- Links
  - Non-functional requirements of low latency and high availability.
  - Data and relationships are not complex.
  - Key-value store.
- Key ranges
  - Favour data consistency as we don't want to accidentally reuse the same key range.
  - Filtering keys by status would help to find available key ranges.
  - Relational database.


**API design:-**
Endpoints:
- createUrl(originalUrl: str) -> shortUrl
  - response code 200, {shortURL: str}
- getLinkHistory(userId: str, sorting: {'asc', 'desc'})
  - response code 200, {links: [str], order: 'asc'}
- redirectURL(shortUrl: str) -> originalUrl
  - response code 200, {originalUrl: str}


**System design:-**
User flows for each of the required functional requirements.
![url_shortener_system.png](../_images/system_design/url_shortener_system.png)


## References
- Udemy course https://www.udemy.com/course/the-bigtech-system-design-interview-bootcamp
- Excalidraw session: https://excalidraw.com/#json=QM7dLZcHbESVnuTPiu06v,pdjPoskF0KknQ6YORHxeHw
- Capacity estimation cheat sheet in _resources folder
- Database book - "Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems" by Martin Kleppmann