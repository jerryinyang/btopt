if order.details.exectype in [
        Order.ExecType.EXIT_LIMIT,
        Order.ExecType.EXIT_STOP,
    ]:
        logger_main.warning(
            f"\n\n\n----- CLOSE ----- \nORDER: {order} | \nPOSITION: {current_position} | \nEXECUTION PRICE: {execution_price}\n\n\n"
        )
    else:
        logger_main.warning(
            f"\n\n\n----- REVERSAL ----- \nORDER: {order} | \nPOSITION: {current_position}\n\n\n"
        )