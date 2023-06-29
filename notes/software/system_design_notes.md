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

Considerations:
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
- Compatibility - Ability of a system to operate seamlessly with othe software, hardware and systems
- Security
  - Basic security measures: TLS encrypted network traffic, API keys for rate limiting

Non-functional requiremetns depend heavily on expected scale. 
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


## References
- Udemy course https://www.udemy.com/course/the-bigtech-system-design-interview-bootcamp
- Exaclidraw session: https://excalidraw.com/#json=QM7dLZcHbESVnuTPiu06v,pdjPoskF0KknQ6YORHxeHw