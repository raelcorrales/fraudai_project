from typing import Dict, Any

class Transaction:
    """
    Clase que representa una transacción bancaria con todos sus detalles.
    Esta clase se utiliza para almacenar y manipular los datos de una transacción,
    y proporciona métodos para convertir los detalles en una cadena para embeddings
    y para convertir los detalles en un diccionario.
    """
    def __init__(self, data: Dict[str, Any]):
        # Asegurarse de que todos los campos existan, incluso si son None o tienen valores por defecto
        self.TransactionID: str = data.get('TransactionID', 'N/A')
        self.AccountID: str = data.get('AccountID', 'N/A')
        self.TransactionAmount: float = float(data.get('TransactionAmount', 0.0))
        self.TransactionDate: str = data.get('TransactionDate', 'N/A')
        self.TransactionType: str = data.get('TransactionType', 'N/A')
        self.Location: str = data.get('Location', 'N/A')
        self.DeviceID: str = data.get('DeviceID', 'N/A')
        self.IPAddress: str = data.get('IP Address', 'N/A') # Ojo con el espacio en el nombre
        self.MerchantID: str = data.get('MerchantID', 'N/A')
        self.Channel: str = data.get('Channel', 'N/A')
        self.CustomerAge: int = int(data.get('CustomerAge', 0))
        self.CustomerOccupation: str = data.get('CustomerOccupation', 'N/A')
        self.TransactionDuration: int = int(data.get('TransactionDuration', 0))
        self.LoginAttempts: int = int(data.get('LoginAttempts', 0))
        self.AccountBalance: float = float(data.get('AccountBalance', 0.0))
        self.PreviousTransactionDate: str = data.get('PreviousTransactionDate', 'N/A')

    def to_string_for_embedding(self) -> str:
        """Convierte los detalles de la transacción en una cadena para el embedding y LLM."""
        return (
            f"Transacción ID: {self.TransactionID}, Cuenta ID: {self.AccountID}, Monto: ${self.TransactionAmount:.2f}, "
            f"Fecha: {self.TransactionDate}, Tipo: {self.TransactionType}, Ubicación: {self.Location}, "
            f"Dispositivo: {self.DeviceID}, IP: {self.IPAddress}, Comercio ID: {self.MerchantID}, "
            f"Canal: {self.Channel}, Edad Cliente: {self.CustomerAge}, Ocupación: {self.CustomerOccupation}, "
            f"Duración Transacción: {self.TransactionDuration}s, Intentos Login: {self.LoginAttempts}, "
            f"Balance Cuenta: ${self.AccountBalance:.2f}, Fecha Transacción Previa: {self.PreviousTransactionDate}."
            f" Contexto: Tipo de transacción {self.TransactionType} en {self.Location}, con canal {self.Channel}. "
            f"Ocupación: {self.CustomerOccupation}, Edad: {self.CustomerAge}. "
        )
    def to_dict(self) -> Dict[str, Any]:
        """Convierte los detalles de la transacción en un diccionario."""
        return {
            'TransactionID': self.TransactionID,
            'AccountID': self.AccountID,
            'TransactionAmount': self.TransactionAmount,
            'TransactionDate': self.TransactionDate,
            'TransactionType': self.TransactionType,
            'Location': self.Location,
            'DeviceID': self.DeviceID,
            'IP Address': self.IPAddress,  # Ojo con el espacio en el nombre
            'MerchantID': self.MerchantID,
            'Channel': self.Channel,
            'CustomerAge': self.CustomerAge,
            'CustomerOccupation': self.CustomerOccupation,
            'TransactionDuration': self.TransactionDuration,
            'LoginAttempts': self.LoginAttempts,
            'AccountBalance': self.AccountBalance,
            'PreviousTransactionDate': self.PreviousTransactionDate
        }
