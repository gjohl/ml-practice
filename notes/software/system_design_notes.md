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

**API design:-**

**System design:-**


## References
- Udemy course https://www.udemy.com/course/the-bigtech-system-design-interview-bootcamp
- Excalidraw session: https://excalidraw.com/#json=QM7dLZcHbESVnuTPiu06v,pdjPoskF0KknQ6YORHxeHw
- Capacity estimation cheat sheet in _resources folder