import logging
logging.basicConfig(level=logging.DEBUG)

from backend.db import create_all, get_session
from data.payment_feed_simulator import PaymentFeedSimulator
from data.debtor_reply_simulator import DebtorReplySimulator
from backend.orchestrator import run_batch_step
from backend.models import Invoice
from datetime import date

def main():
    session = get_session()
    payment_feed = PaymentFeedSimulator()
    reply_sim = DebtorReplySimulator()
    run_batch_step(session, date.today(), payment_feed, reply_sim)

if __name__ == "__main__":
    main()
